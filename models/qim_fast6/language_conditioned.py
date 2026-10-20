"""F5: language-conditioned Tiny Mamba."""

from __future__ import annotations

import torch
from torch import nn

from models.qim_mamba.shared_utils import canonical_geometry_order

from .common import Fast6QIMBase, _text_context, fast_dim, make_fast_stack


class LanguageConditionedMambaMixer(nn.Module):
    def __init__(self, args, dim_in):
        super().__init__()
        width = fast_dim(args)
        self.input_proj = nn.Linear(dim_in, width)
        self.language_proj = nn.Linear(dim_in, width)
        self.mamba = make_fast_stack(args, width)
        self.output_proj = nn.Linear(width, dim_in)
        self.condition_gate = nn.Sequential(
            nn.Linear(dim_in * 2, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(float(getattr(args, "merger_dropout", 0.0)))
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, current, tracks, data):
        context = _text_context(data, current.shape[-1], current.device, current.dtype)
        order = canonical_geometry_order(tracks, current)
        language = self.language_proj(context)[None, None, :]
        tokens = self.input_proj(current[order])[None, :, :] + language
        encoded = self.mamba(tokens).squeeze(0)
        update = self.output_proj(encoded)
        update = update[torch.argsort(order)]
        context_rows = context.expand(current.shape[0], -1)
        gate = self.condition_gate(torch.cat((current, context_rows), dim=-1)).sigmoid()
        return self.norm(current + self.dropout(gate * update))


class LanguageConditionedMambaQIM(Fast6QIMBase):
    variant_name = "FAST6:language_conditioned_mamba"
    requires_qim_text = True

    def _build_variant_layers(self):
        self.mixer = LanguageConditionedMambaMixer(self.args, self.dim_in)

    def _variant_mix(self, tgt, tracks, data):
        return self.mixer(tgt, tracks, data)
