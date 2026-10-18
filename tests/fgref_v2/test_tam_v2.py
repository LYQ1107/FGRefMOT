import pytest
import torch
from torch import nn
from types import SimpleNamespace
from transformers import BatchEncoding

from models.fgref_v2 import (
    ReferringPairHead,
    TextEvidenceEncoder,
    TrackCentricTAMV2,
    TrackEvidenceSampler,
    build_parameter_matched_controls,
)
from models.structures import Instances
from models.transrmot_pro import TransRMOT


def _text_inputs():
    tokens = [["<s>", "blue", "number", "16", "player", "not", "holding", "ball", "right", "unknown", "</s>"]]
    attention = torch.ones(1, len(tokens[0]), dtype=torch.long)
    embeddings = torch.randn(1, len(tokens[0]), 8, requires_grad=True)
    return tokens, attention, embeddings


def test_text_evidence_preserves_full_tokens_and_routes_groups():
    tokens, attention, embeddings = _text_inputs()
    output = TextEvidenceEncoder()(tokens, attention, None, embeddings)
    assert output.full_text_tokens is embeddings
    assert output.identity_token_mask[0, [1, 2, 3, 4, 7]].all()
    assert output.action_token_mask[0, 6]
    assert output.relation_token_mask[0, 8]
    assert output.negation_token_mask[0, 5]
    assert output.unknown_token_mask[0, 9]
    assert output.attribute_presence.shape == (1, 4)
    assert torch.equal(output.valid_token_mask, attention.bool())


def test_sampler_handles_out_of_bounds_boxes_and_keeps_gradient_path():
    sampler = TrackEvidenceSampler([3], 4, num_local=4, num_context=4)
    visual = [torch.randn(1, 3, 6, 8, requires_grad=True)]
    boxes = torch.tensor([
        [[0.08, 0.08, 0.20, 0.20], [1.15, 0.5, 0.2, 0.2]],
    ])
    query = torch.randn(1, 2, 4)
    output = sampler(visual, boxes, query)
    assert output.local_evidence.shape == (1, 2, 4, 4)
    assert output.context_evidence.shape == (1, 2, 4, 4)
    assert output.valid_sample_mask.shape == (1, 2, 8)
    assert output.valid_sample_mask[0, 0].any()
    assert not output.valid_sample_mask[0, 1].any()
    (output.local_evidence.square().mean() + output.context_evidence.square().mean()).backward()
    assert visual[0].grad is not None


def test_tam_uses_predicted_boxes_and_rejects_implicit_diagnostic_boxes():
    encoder = TextEvidenceEncoder()
    sampler = TrackEvidenceSampler([3], 4, num_local=2, num_context=2)
    head = ReferringPairHead(4, 4, num_heads=2)
    tam = TrackCentricTAMV2(encoder, sampler, head)
    tokens, attention, embeddings = _text_inputs()
    embeddings = embeddings[:, : len(tokens[0]), :4].detach().requires_grad_()
    output = tam(
        input_ids=tokens,
        attention_mask=attention,
        offset_mapping=None,
        text_embeddings=embeddings,
        visual_pyramid=[torch.randn(1, 3, 8, 8, requires_grad=True)],
        predicted_boxes=torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]]]),
        queries=torch.randn(1, 2, 4, requires_grad=True),
    )
    assert output.refer_logits.shape == (1, 2, 1)
    assert not torch.allclose(output.refer_logits.sigmoid().sum(dim=1), torch.ones(1))
    with pytest.raises(ValueError, match="diagnostic boxes"):
        tam(
            input_ids=tokens,
            attention_mask=attention,
            offset_mapping=None,
            text_embeddings=embeddings,
            visual_pyramid=[torch.randn(1, 3, 8, 8)],
            predicted_boxes=torch.zeros(1, 2, 4),
            diagnostic_boxes=torch.ones(1, 2, 4),
            queries=torch.randn(1, 2, 4),
        )


def test_parameter_matched_controls_are_both_independent_logit_heads():
    pair, mlp = build_parameter_matched_controls(8, 8, num_heads=2)
    pair_params = sum(p.numel() for p in pair.parameters())
    mlp_params = sum(p.numel() for p in mlp.parameters())
    assert pair_params > 0 and mlp_params > 0
    assert abs(pair_params - mlp_params) / pair_params < 0.35
    queries = torch.randn(2, 3, 8)
    local = torch.randn(2, 3, 2, 8)
    context = torch.randn(2, 3, 2, 8)
    text = torch.randn(2, 4, 8)
    assert pair(queries, local, context, text).shape == (2, 3, 1)
    assert mlp(queries, local, context, text).shape == (2, 3, 1)


def test_transrmot_optional_tam_path_preserves_decoder_referring_score():
    model = object.__new__(TransRMOT)
    nn.Module.__init__(model)
    model.fgref_v2_tam = TrackCentricTAMV2(
        TextEvidenceEncoder(),
        TrackEvidenceSampler([3], 4, num_local=2, num_context=2),
        ReferringPairHead(4, 4, num_heads=2),
    )
    tracks = Instances(
        (32, 32),
        output_embedding=torch.randn(2, 4),
        cache_pred_refers=torch.zeros(2, 1),
    )
    baseline = torch.tensor([[[1.0], [2.0]]])
    frame = {
        "pred_boxes": torch.full((1, 2, 4), 0.5),
        "pred_refers": baseline.clone(),
        "text_word_mask": torch.zeros(1, 5, dtype=torch.bool),
        "fgref_v2_visual_pyramid": [torch.randn(1, 3, 8, 8)],
        "fgref_v2_feature_masks": [torch.zeros(1, 8, 8, dtype=torch.bool)],
        "fgref_v2_input_ids": [["blue", "player", "holding", "ball", "right"]],
        "fgref_v2_offset_mapping": None,
        "fgref_v2_text_embeddings": torch.randn(1, 5, 4),
    }
    result = model._apply_fgref_v2_tam(frame, tracks)
    assert torch.equal(result["pred_refers_baseline"], baseline)
    assert result["pred_refers"].shape == (1, 2, 1)
    assert torch.equal(tracks.cache_pred_refers, result["pred_refers"][0])


def test_forward_text_keeps_offsets_out_of_roberta_inputs():
    model = object.__new__(TransRMOT)
    nn.Module.__init__(model)

    class Tokenizer:
        def batch_encode_plus(self, queries, **kwargs):
            assert kwargs["return_offsets_mapping"] is True
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[0, 42, 2]]),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                    "offset_mapping": torch.tensor([[[0, 0], [0, 4], [5, 11]]]),
                }
            )

    class Encoder(nn.Module):
        def forward(self, input_ids, attention_mask):
            assert tuple(input_ids.shape) == (1, 3)
            assert tuple(attention_mask.shape) == (1, 3)
            return SimpleNamespace(last_hidden_state=torch.randn(1, 3, 4))

    model.tokenizer = Tokenizer()
    model.text_encoder = Encoder()
    model.txt_proj = nn.Identity()
    features, padding, sentence, input_ids, offsets = model.forward_text(
        ["blue player"], torch.device("cpu"), return_tokens=True
    )
    assert features.shape == (1, 3, 4)
    assert sentence.shape == features.shape
    assert padding.shape == (1, 3)
    assert input_ids.shape == (1, 3)
    assert offsets.shape == (1, 3, 2)
