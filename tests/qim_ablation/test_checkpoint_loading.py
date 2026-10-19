import hashlib
import os
from types import SimpleNamespace

import torch

from models.qim_pro import QueryInteractionModule


CHECKPOINT = '/data1/liuyeqiang/fgref_v2_runs/checkpoints/checkpoint0089.pth'
EXPECTED_SHA256 = 'adb2ad9591a4a14dcdd05718b271d6c7b391e97bd9cb20f4fd8e3bfe26e54038'


def test_original_mamba_qim_strict_checkpoint_loading():
    if not os.path.exists(CHECKPOINT):
        return
    digest = hashlib.sha256()
    with open(CHECKPOINT, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == EXPECTED_SHA256
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    source = {k[len('track_embed.'):]: v for k, v in checkpoint['model'].items()
              if k.startswith('track_embed.')}
    module = QueryInteractionModule(
        SimpleNamespace(random_drop=0.1, fp_ratio=0.3,
                        update_query_pos=True, merger_dropout=0.0),
        256, 1024, 512,
    )
    missing, unexpected = module.load_state_dict(source, strict=True)
    assert not missing
    assert not unexpected
