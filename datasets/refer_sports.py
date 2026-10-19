# ------------------------------------------------------------------------
# Refer-Sports adapter for the original FGRefMOT training tree.
#
# The source dataset is MOT-style SportsMOT (img1/ + gt/gt.txt), while the
# original loader expects one normalized label file per image.  This adapter
# keeps the original clip/training interface and converts the source format at
# read time.  It intentionally does not import any FGRefMOT-v2 TAM/QIM code.
# ------------------------------------------------------------------------
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image

import datasets.transforms as T
from models.structures import Instances


_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _numeric_frame_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.name


class SportsMOTDetection:
    """Same-video clip sampler and expression-aware SportsMOT reader.

    ``unknown_mask`` is the protocol-safe default.  SportsMOT expressions only
    provide positive target IDs; absent IDs are UNKNOWN rather than explicit
    negatives.  In that mode the referring loss is valid only for positive
    target instances.  ``legacy_binary`` is available solely for a controlled
    compatibility comparison with the historical binary ``is_ref`` rule.
    """

    def __init__(self, args, image_set, dataset2transform):
        self.args = args
        self.image_set = image_set
        self.split = getattr(args, 'sports_split', image_set)
        if image_set == 'train' and self.split not in {'train', 'val', 'test'}:
            raise ValueError('invalid SportsMOT split: {}'.format(self.split))
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        self.current_epoch = 0
        self.period_idx = 0
        self.ref_protocol = getattr(args, 'sports_ref_protocol', 'unknown_mask')
        if self.ref_protocol not in {'unknown_mask', 'legacy_binary'}:
            raise ValueError('invalid sports_ref_protocol: {}'.format(self.ref_protocol))

        publish_root = Path(args.sportsmot_root)
        self.dataset_root = publish_root / 'dataset'
        self.expression_root = Path(args.sports_expression_root)
        split_file = publish_root / 'splits_txt' / '{}.txt'.format(self.split)
        if not split_file.is_file():
            raise FileNotFoundError('SportsMOT split file does not exist: {}'.format(split_file))
        if not self.dataset_root.is_dir():
            raise FileNotFoundError('SportsMOT dataset root does not exist: {}'.format(self.dataset_root))
        if not self.expression_root.is_dir():
            raise FileNotFoundError('SportsMOT expression root does not exist: {}'.format(self.expression_root))

        self.manifest_records = []
        manifest_path = getattr(args, 'sports_manifest', None)
        if manifest_path and Path(manifest_path).is_file():
            with Path(manifest_path).open() as handle:
                manifest = json.load(handle)
            self.manifest_records = [
                record for record in manifest.get('records', [])
                if record.get('original_split') == self.split
            ]
        if self.manifest_records:
            self.video_names = list(dict.fromkeys(
                record['original_sportsmot_id'] for record in self.manifest_records
            ))
        else:
            self.video_names = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]
        video_list = getattr(args, 'sports_video_list', None)
        if video_list:
            list_path = Path(video_list)
            if not list_path.is_file():
                raise FileNotFoundError('SportsMOT video list does not exist: {}'.format(list_path))
            requested = [x.strip() for x in list_path.read_text().splitlines() if x.strip()]
            missing = sorted(set(requested) - set(self.video_names))
            if missing:
                raise ValueError('video list contains IDs outside the selected manifest split: {}'.format(missing))
            self.video_names = requested
        self.video_dict = {video: i for i, video in enumerate(self.video_names)}
        self.video_to_indices = {}
        self.index_to_video = {}
        self.index_to_local = {}
        self.frame_ids = {}
        self.img_files = []
        self.video_gt = {}
        self.video_expression_paths = {}

        for video_name in self.video_names:
            split_candidates = [self.split, 'train', 'val', 'test']
            video_dir = next(
                (self.dataset_root / candidate / video_name for candidate in split_candidates
                 if (self.dataset_root / candidate / video_name).is_dir()),
                self.dataset_root / self.split / video_name,
            )
            img_dir = video_dir / 'img1'
            gt_path = video_dir / 'gt' / 'gt.txt'
            if not img_dir.is_dir():
                raise FileNotFoundError('missing SportsMOT image directory: {}'.format(img_dir))
            if not gt_path.is_file() and self.split != 'test':
                raise FileNotFoundError('missing SportsMOT ground truth: {}'.format(gt_path))
            expression_paths = sorted((self.expression_root / video_name).glob('*.json'))
            if not expression_paths:
                raise FileNotFoundError('missing expression JSONs for {}'.format(video_name))
            self.video_expression_paths[video_name] = expression_paths
            labels_dir = self.dataset_root / 'labels_with_ids' / video_name
            self.video_gt[video_name] = self._read_global_labels(labels_dir) if labels_dir.is_dir() else {}
            if self.split != 'test' and not self.video_gt[video_name]:
                raise FileNotFoundError(
                    'missing global labels_with_ids for {} (the MOT gt IDs are not the Refer-Sports global namespace)'.format(
                        video_name
                    )
                )

            video_indices = []
            image_paths = sorted(
                [p for p in img_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS],
                key=_numeric_frame_key,
            )
            if not image_paths:
                raise ValueError('no image frames found in {}'.format(img_dir))
            for image_path in image_paths:
                index = len(self.img_files)
                video_indices.append(index)
                self.img_files.append(str(image_path))
                self.index_to_video[index] = video_name
                self.index_to_local[index] = len(video_indices) - 1
                self.frame_ids[index] = int(image_path.stem)
            self.video_to_indices[video_name] = video_indices

        self.lengths = args.sampler_lengths
        self.sampler_steps = args.sampler_steps
        print('SportsMOT split={} videos={} frames={} sampler_steps={} lengths={}'.format(
            self.split, len(self.video_names), len(self.img_files), self.sampler_steps, self.lengths
        ))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            assert len(self.lengths) == len(self.sampler_steps) + 1
            self.num_frames_per_batch = self.lengths[0]
        self._refresh_valid_starts()

    @staticmethod
    def _read_global_labels(labels_dir):
        records = {}
        if not labels_dir.is_dir():
            return records
        for label_path in labels_dir.glob('*.txt'):
            try:
                frame_id = int(label_path.stem)
            except ValueError:
                continue
            rows = []
            with label_path.open() as handle:
              for line_no, raw in enumerate(handle, 1):
                line = raw.strip()
                if not line:
                    continue
                fields = line.replace(',', ' ').split()
                if len(fields) < 6:
                    raise ValueError('invalid labels_with_ids row {}:{}: {}'.format(label_path, line_no, raw.rstrip()))
                try:
                    object_id = int(float(fields[1]))
                    cx, cy, w, h = (float(fields[i]) for i in range(2, 6))
                except ValueError as exc:
                    raise ValueError('invalid labels_with_ids row {}:{}: {}'.format(label_path, line_no, raw.rstrip())) from exc
                if w <= 0 or h <= 0:
                    continue
                rows.append((object_id, cx, cy, w, h))
            records[frame_id] = rows
        return records

    def _refresh_valid_starts(self):
        max_interval = max(1, int(self.sample_interval))
        self._valid_starts = []
        for video_name in self.video_names:
            indices = self.video_to_indices[video_name]
            last_start = len(indices) - 1 - (self.num_frames_per_batch - 1) * max_interval
            for local_start in range(max(0, last_start + 1)):
                self._valid_starts.append(indices[local_start])
        if not self._valid_starts:
            raise ValueError('no valid same-video SportsMOT clips')
        self.item_num = len(self._valid_starts)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            return
        self.period_idx = 0
        for step in self.sampler_steps:
            if epoch >= step:
                self.period_idx += 1
        self.num_frames_per_batch = self.lengths[self.period_idx]
        self._refresh_valid_starts()
        print('SportsMOT set epoch={} period_idx={} clip_length={}'.format(
            epoch, self.period_idx, self.num_frames_per_batch
        ))

    def step_epoch(self):
        self.set_epoch(self.current_epoch + 1)

    def _get_sample_range(self, start_idx):
        if self.sample_mode not in {'fixed_interval', 'random_interval'}:
            raise ValueError('invalid sample mode: {}'.format(self.sample_mode))
        if start_idx < 0 or start_idx >= len(self._valid_starts):
            raise IndexError('sample index {} outside {}'.format(start_idx, len(self._valid_starts)))
        global_start = self._valid_starts[start_idx]
        video_name = self.index_to_video[global_start]
        video_indices = self.video_to_indices[video_name]
        local_start = self.index_to_local[global_start]
        interval = self.sample_interval
        if self.sample_mode == 'random_interval':
            interval = np.random.randint(1, self.sample_interval + 1)
        local_indices = list(video_indices[
            local_start:local_start + (self.num_frames_per_batch - 1) * interval + 1:interval
        ])
        if len(local_indices) != self.num_frames_per_batch:
            raise RuntimeError('invalid same-video SportsMOT clip for {}'.format(video_name))
        return local_indices

    def _expression_for_sample(self, sample_idx, video_name):
        paths = self.video_expression_paths[video_name]
        # Deterministic per epoch/index selection keeps all QIM variants on the
        # same expression sequence while still covering multiple expressions.
        path = paths[(sample_idx + 9973 * self.current_epoch) % len(paths)]
        with path.open() as handle:
            expression = json.load(handle)
        if expression.get('video_name') != video_name:
            raise ValueError('expression/video mismatch: {} vs {}'.format(expression.get('video_name'), video_name))
        expression['_expression_id'] = '{}:{}'.format(video_name, path.stem)
        return expression

    @staticmethod
    def _targets_to_instances(targets, img_shape):
        instances = Instances(tuple(img_shape))
        for key in ('boxes', 'labels', 'obj_ids', 'area', 'is_ref', 'is_ref_valid'):
            setattr(instances, key, targets[key])
        return instances

    def _pre_single_frame(self, index, expression):
        image_path = self.img_files[index]
        video_name = self.index_to_video[index]
        frame_id = self.frame_ids[index]
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        rows = self.video_gt[video_name].get(frame_id, [])
        label_map = expression.get('label', {})
        positive_ids = {int(x) for x in label_map.get(str(frame_id), [])}
        frame_has_expression = str(frame_id) in label_map

        boxes = []
        areas = []
        object_ids = []
        is_ref = []
        is_ref_valid = []
        offset = self.video_dict[video_name] * 1000000
        for object_id, cx, cy, box_w, box_h in rows:
            x = width * (cx - box_w / 2.0)
            y = height * (cy - box_h / 2.0)
            w = width * box_w
            h = height * box_h
            boxes.append([x, y, x + w, y + h])
            areas.append(w * h)
            object_ids.append(object_id + offset if object_id >= 0 else object_id)
            positive = object_id in positive_ids
            is_ref.append(float(positive))
            # The recovered Refer-Sports manifest defines absent positives as
            # UNKNOWN.  Only an explicitly positive target is valid by default.
            is_ref_valid.append(bool(positive) if self.ref_protocol == 'unknown_mask' else frame_has_expression)

        targets = {
            'dataset': 'SportsMOT',
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'labels': torch.zeros(len(boxes), dtype=torch.int64),
            'obj_ids': torch.as_tensor(object_ids, dtype=torch.int64),
            'is_ref': torch.as_tensor(is_ref, dtype=torch.float32),
            'is_ref_valid': torch.as_tensor(is_ref_valid, dtype=torch.bool),
            'image_id': torch.as_tensor(index),
            'frame_id': frame_id,
            'video_name': video_name,
            'size': torch.as_tensor([height, width]),
            'orig_size': torch.as_tensor([height, width]),
        }
        return img, targets

    def __getitem__(self, sample_idx):
        sample_indices = self._get_sample_range(sample_idx)
        video_name = self.index_to_video[sample_indices[0]]
        expression = self._expression_for_sample(sample_idx, video_name)
        images, targets = [], []
        for index in sample_indices:
            image, target = self._pre_single_frame(index, expression)
            images.append(image)
            targets.append(target)

        transform = self.dataset2transform['SportsMOT']
        if transform is not None:
            images, targets = transform(images, targets)
        gt_instances = [
            self._targets_to_instances(target, image.shape[1:3])
            for image, target in zip(images, targets)
        ]
        return {
            'imgs': images,
            'sentences': [expression['sentence']],
            'gt_instances': gt_instances,
            'dataset_name': 'SportsMOT',
        }

    def __len__(self):
        return self.item_num


def make_transforms_for_sports(image_set, args=None):
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ]),
            ),
            normalize,
        ])
    if image_set in {'val', 'test'}:
        return T.MotCompose([T.MotRandomResize([800], max_size=1333), normalize])
    raise ValueError('unknown SportsMOT image set: {}'.format(image_set))


def build(image_set, args):
    split = getattr(args, 'sports_split', image_set)
    transform = make_transforms_for_sports(image_set, args)
    dataset2transform = {'SportsMOT': transform}
    return SportsMOTDetection(args, image_set=image_set, dataset2transform=dataset2transform)
