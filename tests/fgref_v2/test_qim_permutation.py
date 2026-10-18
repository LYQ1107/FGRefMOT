import ast
from pathlib import Path

import torch

from models.structures import Instances


QIM_SOURCE = Path(__file__).parents[2] / "models" / "qim_pro.py"


def _tracks():
    return Instances(
        (64, 64),
        obj_idxes=torch.tensor([17, 4, 29]),
        scores=torch.ones(3),
        iou=torch.ones(3),
        output_embedding=torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        ),
    )


def test_instances_boolean_selection_preserves_track_id_alignment():
    tracks = _tracks()
    active = tracks[torch.tensor([True, False, True])]
    assert active.obj_idxes.tolist() == [17, 29]
    assert active.output_embedding[:, 0].tolist() == [1.0, 3.0]


def test_current_qim_uses_track_rows_as_the_mamba_sequence():
    source = QIM_SOURCE.read_text()
    assert source.count("tgt_expanded = tgt.unsqueeze(0)") >= 2
    assert source.count("inference_params = None") >= 2

    # This is the shape contract of the current call: [batch=1, active_rows, D].
    # A sequence mixer can therefore couple different physical tracks.
    tracks = _tracks()
    sequence = tracks.output_embedding.unsqueeze(0)
    reordered = tracks[torch.tensor([2, 0, 1])].output_embedding.unsqueeze(0)
    original_context = sequence.cumsum(dim=1)
    reordered_context = reordered.cumsum(dim=1)
    inverse = torch.tensor([1, 2, 0])
    restored = reordered_context[:, inverse]
    assert not torch.equal(original_context, restored)


def test_qim_source_has_no_persistent_track_keyed_mamba_state():
    tree = ast.parse(QIM_SOURCE.read_text())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "inference_params" in names
    assert "track_memory" not in names
    assert "history_state" not in names
