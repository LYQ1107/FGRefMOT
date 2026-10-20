"""Leakage-safe YOLOX-X detector experiment for the McByte tracker.

The data root contains SportsMOT train with the 19 Refer-Sports evaluation
videos removed from ``clean_train.json``.  The detector is initialized from
the generic COCO YOLOX-X checkpoint and never from a SportsMOT checkpoint.
"""

import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = "mcbyte_yolox_x_loo19"
        seed_text = os.environ.get("RMOT_MCBYTE_SEED", "42").strip().lower()
        self.seed = None if seed_text in {"", "none", "null"} else int(seed_text)

        # All paths can be overridden by the launcher environment.
        self.data_root = os.environ.get(
            "RMOT_MCBYTE_DETECTOR_DATA",
            "/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_leakage_safe_detector_loo19_20261020",
        )
        self.output_dir = os.environ.get(
            "RMOT_MCBYTE_DETECTOR_OUTPUT",
            "/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/mcbyte_yolox_x_loo19_train_20261020",
        )
        self.train_ann = "clean_train.json"
        self.val_ann = "val.json"

        size_text = os.environ.get("RMOT_MCBYTE_INPUT_SIZE", "800x1440")
        try:
            size = tuple(int(part) for part in size_text.lower().split("x", 1))
            if len(size) != 2:
                raise ValueError
        except ValueError as exc:
            raise ValueError("RMOT_MCBYTE_INPUT_SIZE must be HxW, e.g. 640x1152") from exc
        self.input_size = size
        self.test_size = size
        self.random_size = None if os.environ.get("RMOT_MCBYTE_DISABLE_RANDOM_RESIZE", "0") == "1" else (18, 32)
        self.max_epoch = 80
        self.no_aug_epochs = 10
        self.eval_interval = 5
        self.print_interval = 20
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.data_num_workers = 4
        self.enable_mixup = os.environ.get("RMOT_MCBYTE_DISABLE_MIXUP", "0") != "1"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MOTDataset,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
        )

        dataset = MOTDataset(
            data_dir=self.data_root,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=600,
            ),
        )
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1200,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        return DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        dataset = MOTDataset(
            data_dir=self.data_root,
            json_file=self.val_ann,
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        return COCOEvaluator(
            dataloader=loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
