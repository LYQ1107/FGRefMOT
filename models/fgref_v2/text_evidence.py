"""Token-level evidence groups for fine-grained referring expressions.

This first implementation is intentionally a small, inspectable rule router.
It never removes tokens.  Unrecognised words are represented by
``unknown_token_mask`` and remain in ``full_text_tokens`` so later learned
routing can use them.
"""

from dataclasses import dataclass
import re
from typing import Any, Iterable, List, Optional, Sequence

import torch
from torch import Tensor, nn


@dataclass
class TextEvidence:
    """Structured token evidence returned by :class:`TextEvidenceEncoder`.

    All masks have shape ``[B, L]`` and use ``True`` for a selected/valid
    token.  ``attribute_presence`` has four columns in the order
    ``identity, action, relation, unknown``.
    """

    full_text_tokens: Tensor
    identity_token_mask: Tensor
    action_token_mask: Tensor
    relation_token_mask: Tensor
    attribute_presence: Tensor
    unknown_token_mask: Tensor
    negation_token_mask: Tensor
    attention_mask: Tensor

    @property
    def valid_token_mask(self) -> Tensor:
        return self.attention_mask


class TextEvidenceEncoder(nn.Module):
    """Map a tokenized expression to lightweight semantic evidence groups.

    ``input_ids`` may be integer token IDs (in which case a tokenizer is
    needed for useful lexical routing) or a nested sequence of token strings,
    which is convenient for unit tests.  ``raw_text`` is optional and can be a
    string or a list of strings; when supplied with tokenizer offsets it lets
    the router handle subword pieces and compound descriptions.

    The lexical rules are a diagnostic starting point, not a claim that the
    groups are ground-truth labels.  They are kept in code so every routed
    token can be audited and later replaced by a learned router in an explicit
    ablation.
    """

    DEFAULT_IDENTITY_TERMS = {
        "person",
        "people",
        "player",
        "players",
        "man",
        "woman",
        "men",
        "women",
        "boy",
        "girl",
        "ball",
        "shirt",
        "jersey",
        "uniform",
        "number",
        "no",
        "color",
        "colour",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "white",
        "black",
        "purple",
        "pink",
        "brown",
        "gray",
        "grey",
    }
    DEFAULT_ACTION_TERMS = {
        "hold",
        "holds",
        "holding",
        "carry",
        "carries",
        "carrying",
        "run",
        "runs",
        "running",
        "walk",
        "walks",
        "walking",
        "stand",
        "stands",
        "standing",
        "sit",
        "sits",
        "sitting",
        "jump",
        "jumps",
        "jumping",
        "kick",
        "kicks",
        "kicking",
        "throw",
        "throws",
        "throwing",
        "serve",
        "serves",
        "serving",
        "move",
        "moving",
    }
    DEFAULT_RELATION_TERMS = {
        "left",
        "right",
        "front",
        "behind",
        "back",
        "near",
        "next",
        "beside",
        "between",
        "center",
        "middle",
        "top",
        "bottom",
        "above",
        "below",
        "inside",
        "outside",
        "under",
        "over",
    }
    DEFAULT_NEGATION_TERMS = {"not", "no", "never", "without", "neither"}

    _SPECIAL_TOKENS = {
        "s",
        "/s",
        "pad",
        "cls",
        "sep",
        "bos",
        "eos",
    }

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        identity_terms: Optional[Iterable[str]] = None,
        action_terms: Optional[Iterable[str]] = None,
        relation_terms: Optional[Iterable[str]] = None,
        negation_terms: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.identity_terms = self._normalise_terms(
            identity_terms or self.DEFAULT_IDENTITY_TERMS
        )
        self.action_terms = self._normalise_terms(
            action_terms or self.DEFAULT_ACTION_TERMS
        )
        self.relation_terms = self._normalise_terms(
            relation_terms or self.DEFAULT_RELATION_TERMS
        )
        self.negation_terms = self._normalise_terms(
            negation_terms or self.DEFAULT_NEGATION_TERMS
        )

    @staticmethod
    def _normalise_terms(terms: Iterable[str]) -> set[str]:
        return {TextEvidenceEncoder._normalise_token(str(term)) for term in terms}

    @staticmethod
    def _normalise_token(token: str) -> str:
        token = token.strip().lower()
        token = token.replace("##", "").replace("Ġ", "").replace("▁", "")
        token = token.replace("byte_encoder", "")
        token = re.sub(r"[^a-z0-9_-]", "", token)
        return token

    @staticmethod
    def _as_rows(value: Any, batch: int, length: int) -> List[List[Any]]:
        if value is None:
            return [["" for _ in range(length)] for _ in range(batch)]
        if isinstance(value, Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, (tuple, list)):
            if len(value) == 0:
                return [["" for _ in range(length)] for _ in range(batch)]
            if len(value) == batch and isinstance(value[0], (tuple, list, Tensor)):
                rows = [list(row) for row in value]
            else:
                rows = [list(value)]
        else:
            rows = [[value]]
        if len(rows) == 1 and batch > 1:
            rows = rows * batch
        if len(rows) != batch:
            raise ValueError(f"expected {batch} rows, got {len(rows)}")
        return [row[:length] + [""] * max(0, length - len(row)) for row in rows]

    def _token_strings(
        self,
        input_ids: Any,
        batch: int,
        length: int,
        offset_mapping: Optional[Any],
        raw_text: Optional[Any],
    ) -> List[List[str]]:
        rows = self._as_rows(input_ids, batch, length)
        if self.tokenizer is not None and isinstance(input_ids, Tensor):
            converted: List[List[str]] = []
            for row in rows:
                try:
                    tokens = self.tokenizer.convert_ids_to_tokens(row)
                except Exception:
                    tokens = [str(item) for item in row]
                converted.append([str(item) for item in tokens])
            rows = converted

        # Offsets are preferable for expressions with compound words or BPE
        # fragments. They do not remove any token; a zero-length special token
        # simply remains an unselected full-text token.
        if offset_mapping is not None and raw_text is not None:
            offsets = self._as_rows(offset_mapping, batch, length)
            texts = raw_text if isinstance(raw_text, (list, tuple)) else [raw_text] * batch
            if len(texts) == 1 and batch > 1:
                texts = list(texts) * batch
            for b in range(batch):
                source = str(texts[b])
                for i, item in enumerate(offsets[b]):
                    if isinstance(item, Tensor):
                        item = item.tolist()
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        continue
                    start, end = int(item[0]), int(item[1])
                    if 0 <= start <= end <= len(source) and end > start:
                        rows[b][i] = source[start:end]
        return [[str(token) for token in row] for row in rows]

    def forward(
        self,
        input_ids: Any,
        attention_mask: Tensor,
        offset_mapping: Optional[Any],
        text_embeddings: Tensor,
        *,
        raw_text: Optional[Any] = None,
    ) -> TextEvidence:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, L]")
        if text_embeddings.ndim != 3:
            raise ValueError("text_embeddings must have shape [B, L, D] or [L, B, D]")
        batch, length = attention_mask.shape
        if text_embeddings.shape[:2] == (length, batch):
            text_embeddings = text_embeddings.transpose(0, 1)
        if text_embeddings.shape[:2] != (batch, length):
            raise ValueError(
                f"text_embeddings leading shape {tuple(text_embeddings.shape[:2])} "
                f"does not match {(batch, length)}"
            )

        tokens = self._token_strings(
            input_ids, batch, length, offset_mapping, raw_text
        )
        device = text_embeddings.device
        valid = attention_mask.to(device=device, dtype=torch.bool)
        identity = torch.zeros((batch, length), dtype=torch.bool, device=device)
        action = torch.zeros_like(identity)
        relation = torch.zeros_like(identity)
        unknown = torch.zeros_like(identity)
        negation = torch.zeros_like(identity)

        for b in range(batch):
            for i, token in enumerate(tokens[b]):
                if not bool(valid[b, i]):
                    continue
                normalised = self._normalise_token(token)
                if not normalised or normalised in self._SPECIAL_TOKENS:
                    continue
                has_digit = bool(re.search(r"\d", normalised))
                if normalised in self.negation_terms:
                    negation[b, i] = True
                if (
                    normalised in self.identity_terms
                    or has_digit
                    or any(term in normalised for term in ("jersey", "uniform"))
                ):
                    identity[b, i] = True
                if normalised in self.action_terms or any(
                    normalised.startswith(stem)
                    for stem in ("hold", "carry", "run", "walk", "stand", "jump", "kick", "throw", "serv")
                ):
                    action[b, i] = True
                if normalised in self.relation_terms:
                    relation[b, i] = True
                if not (identity[b, i] or action[b, i] or relation[b, i]):
                    unknown[b, i] = True

        presence = torch.stack(
            [identity.any(dim=1), action.any(dim=1), relation.any(dim=1), unknown.any(dim=1)],
            dim=-1,
        ).to(dtype=text_embeddings.dtype)
        return TextEvidence(
            full_text_tokens=text_embeddings,
            identity_token_mask=identity,
            action_token_mask=action,
            relation_token_mask=relation,
            attribute_presence=presence,
            unknown_token_mask=unknown,
            negation_token_mask=negation,
            attention_mask=valid,
        )
