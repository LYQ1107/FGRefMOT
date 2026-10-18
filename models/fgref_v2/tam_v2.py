"""Track-centric Fine-Grained Evidence Alignment (TAM-v2)."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from .referring_pair_head import ReferringPairHead
from .text_evidence import TextEvidence, TextEvidenceEncoder
from .track_evidence_sampler import TrackEvidenceSampler


@dataclass
class TAMV2Output:
    refer_logits: Tensor
    local_evidence: Tensor
    context_evidence: Tensor
    valid_sample_mask: Tensor
    text_evidence: TextEvidence


class TrackCentricTAMV2(nn.Module):
    """A standalone TAM-v2 referring path.

    The formal path always consumes ``predicted_boxes``.  A separate box
    argument is accepted only when ``diagnostic=True`` and
    ``allow_box_diagnostic=True``; this makes a GT-box diagnostic impossible
    to use accidentally as an online result.
    """

    def __init__(
        self,
        text_evidence_encoder: TextEvidenceEncoder,
        sampler: TrackEvidenceSampler,
        referring_head: ReferringPairHead,
        *,
        allow_box_diagnostic: bool = False,
    ) -> None:
        super().__init__()
        self.text_evidence_encoder = text_evidence_encoder
        self.sampler = sampler
        self.referring_head = referring_head
        self.allow_box_diagnostic = bool(allow_box_diagnostic)

    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Tensor,
        offset_mapping: Optional[Any],
        text_embeddings: Tensor,
        visual_pyramid: Sequence[Tensor],
        predicted_boxes: Tensor,
        queries: Tensor,
        history_features: Optional[Tensor] = None,
        feature_masks: Optional[Sequence[Tensor]] = None,
        raw_text: Optional[Any] = None,
        diagnostic_boxes: Optional[Tensor] = None,
        diagnostic: bool = False,
    ) -> TAMV2Output:
        if predicted_boxes is None:
            raise ValueError("predicted_boxes are required for the formal TAM path")
        if diagnostic_boxes is not None:
            if not diagnostic or not self.allow_box_diagnostic:
                raise ValueError(
                    "diagnostic boxes require diagnostic=True and explicit allow_box_diagnostic"
                )
            boxes = diagnostic_boxes
        else:
            boxes = predicted_boxes

        text_evidence = self.text_evidence_encoder(
            input_ids,
            attention_mask,
            offset_mapping,
            text_embeddings,
            raw_text=raw_text,
        )
        sampled = self.sampler(
            visual_pyramid,
            boxes,
            queries,
            text_condition=text_evidence.full_text_tokens,
            text_padding_mask=~attention_mask.to(dtype=torch.bool, device=queries.device),
            feature_masks=feature_masks,
        )
        local_valid = sampled.valid_sample_mask[:, :, : self.sampler.num_local]
        context_valid = sampled.valid_sample_mask[:, :, self.sampler.num_local :]
        refer_logits = self.referring_head(
            queries,
            sampled.local_evidence,
            sampled.context_evidence,
            text_evidence.full_text_tokens,
            history_features,
            local_valid_mask=local_valid,
            context_valid_mask=context_valid,
            text_valid_mask=attention_mask.to(dtype=torch.bool, device=queries.device),
        )
        return TAMV2Output(
            refer_logits=refer_logits,
            local_evidence=sampled.local_evidence,
            context_evidence=sampled.context_evidence,
            valid_sample_mask=sampled.valid_sample_mask,
            text_evidence=text_evidence,
        )
