"""F3: stateful causal Mamba with one persistent state per active track."""

from __future__ import annotations

import torch
from torch import nn

from .common import Fast6QIMBase, fast_dim, make_fast_stack
from .state_manager import ensure_bool_field, ensure_tensor_field


class StreamingStateMambaMixer(nn.Module):
    requires_qim_state = True

    def __init__(self, args, dim_in):
        super().__init__()
        self.state_dim = fast_dim(args)
        self.input_proj = nn.Linear(dim_in, self.state_dim)
        self.mamba = make_fast_stack(args, self.state_dim)
        self.output_proj = nn.Linear(self.state_dim, dim_in)
        self.dropout = nn.Dropout(float(getattr(args, "merger_dropout", 0.0)))
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, current, tracks):
        number = current.shape[0]
        if number == 0:
            return current
        state = ensure_tensor_field(
            tracks, "qim_stream_state", (number, self.state_dim), dtype=current.dtype
        ).to(dtype=current.dtype)
        valid = ensure_bool_field(tracks, "qim_stream_state_valid")
        previous = torch.where(valid[:, None], state, torch.zeros_like(state))
        tokens = torch.stack((previous, self.input_proj(current)), dim=1)
        encoded = self.mamba(tokens)
        new_state = encoded[:, 1]
        tracks.qim_stream_state = new_state
        tracks.qim_stream_state_valid = torch.ones_like(valid)
        update = self.output_proj(new_state)
        return self.norm(current + self.dropout(update))


class StreamingStateMambaQIM(Fast6QIMBase):
    variant_name = "FAST6:streaming_state_mamba"
    requires_qim_state = True

    def _build_variant_layers(self):
        self.mixer = StreamingStateMambaMixer(self.args, self.dim_in)
        self.state_dim = self.mixer.state_dim

    def _variant_mix(self, tgt, tracks, data):
        return self.mixer(tgt, tracks)
