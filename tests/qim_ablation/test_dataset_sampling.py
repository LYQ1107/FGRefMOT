import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from datasets.refer_kitti import DetMOTDetection


def test_sample_starts_never_cross_video_boundary(tmp_path):
    root = tmp_path / 'KITTI'
    expression_root = tmp_path / 'expression'
    list_path = tmp_path / 'train.txt'
    lines = []
    for video in ('video_a', 'video_b'):
        image_dir = root / 'training' / 'image_02' / video
        label_dir = root / 'labels_with_ids' / 'image_02' / video
        expression_dir = expression_root / video
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        expression_dir.mkdir(parents=True)
        with open(expression_dir / 'expr.json', 'w', encoding='utf-8') as handle:
            json.dump({'sentence': 'player', 'label': {}, 'ignore': {}}, handle)
        for frame in range(1, 6):
            name = f'{frame:06d}.png'
            Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(image_dir / name)
            (label_dir / name.replace('.png', '.txt')).write_text(
                '0 1 0.5 0.5 0.2 0.2\n', encoding='utf-8'
            )
            lines.append(f'training/image_02/{video}/{name}\n')
    list_path.write_text(''.join(lines), encoding='utf-8')

    args = SimpleNamespace(
        sampler_lengths=[4], sampler_steps=[], sample_mode='fixed_interval',
        sample_interval=1, vis=False, rmot_path=str(tmp_path),
    )
    dataset = DetMOTDetection(args, str(list_path), root, {'KITTI': None})
    assert len(dataset) == 4
    for index in range(len(dataset)):
        sample_indices = dataset._get_sample_range(index)
        videos = {dataset.index_to_video[item] for item in sample_indices}
        assert len(videos) == 1
