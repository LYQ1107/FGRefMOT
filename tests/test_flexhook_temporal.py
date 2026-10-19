import unittest

import torch

from adapters.flexhook_temporal import (
    EvidenceBuffer,
    ReliabilityGate,
    TemporalEvidenceAdapter,
    TemporalReferringHead,
    run_causal_sequence,
)


class FlexHookTemporalTest(unittest.TestCase):
    def test_buffer_is_causal_and_expression_scoped(self):
        buffer = EvidenceBuffer(feature_dim=4, dynamic_window=3, identity_momentum=0.5)
        first = buffer.update(
            namespace=("video", "blue jersey"),
            track_id=7,
            frame_index=0,
            feature=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            bbox=torch.tensor([0.1, 0.1, 0.2, 0.2]),
        )
        other_expression = buffer.update(
            namespace=("video", "holding ball"),
            track_id=7,
            frame_index=0,
            feature=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            bbox=torch.tensor([0.1, 0.1, 0.2, 0.2]),
        )
        self.assertTrue(torch.allclose(first.identity, torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(other_expression.identity, torch.tensor([0.0, 1.0, 0.0, 0.0])))

        second = buffer.update(
            namespace=("video", "blue jersey"),
            track_id=7,
            frame_index=1,
            feature=torch.tensor([0.0, 0.0, 1.0, 0.0]),
            bbox=torch.tensor([0.2, 0.1, 0.2, 0.2]),
        )
        self.assertEqual(second.dynamic_history.shape, (2, 4))
        self.assertTrue(torch.allclose(second.velocity, torch.tensor([0.1, 0.0, 0.0, 0.0])))
        with self.assertRaises(ValueError):
            buffer.update(
                namespace=("video", "blue jersey"),
                track_id=7,
                frame_index=1,
                feature=torch.ones(4),
                bbox=torch.zeros(4),
            )

    def test_conflicting_observation_does_not_overwrite_identity(self):
        buffer = EvidenceBuffer(feature_dim=4, identity_momentum=0.5)
        first = buffer.update(
            namespace="sequence/expression",
            track_id=1,
            frame_index=0,
            feature=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            bbox=torch.zeros(4),
        )
        second = buffer.update(
            namespace="sequence/expression",
            track_id=1,
            frame_index=1,
            feature=torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            bbox=torch.zeros(4),
            detection_confidence=0.1,
            visibility=0.1,
            local_quality=0.1,
            candidate_conflict=1.0,
        )
        self.assertTrue(first.wrote_identity)
        self.assertFalse(second.wrote_identity)
        self.assertTrue(torch.allclose(second.identity, first.identity))
        self.assertLess(float(second.reliability), 0.5)

    def test_reliability_gate_does_not_take_referring_score(self):
        gate = ReliabilityGate()
        score = gate(1.0, 1.0, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(float(score), 1.0, places=6)

    def test_all_variants_return_independent_candidate_logits(self):
        torch.manual_seed(4)
        current = torch.randn(2, 3, 8)
        language = torch.randn(2, 8)
        identity = torch.randn(2, 3, 8)
        dynamic = torch.randn(2, 3, 8)
        history = torch.randn(2, 3, 4, 8)
        reliability = torch.full((2, 3), 0.8)
        for variant in [f"H{i}" for i in range(9)]:
            head = TemporalReferringHead(8, variant=variant)
            logits = head(
                current,
                language,
                identity=identity,
                dynamic=dynamic,
                dynamic_history=history,
                reliability=reliability,
            )
            self.assertEqual(logits.shape, (2, 3, 1), variant)
            self.assertTrue(torch.isfinite(logits).all())
            # A candidate-independent softmax would force a normalization over
            # N.  This head emits one Bernoulli logit per candidate instead.
            self.assertFalse(torch.allclose(logits.sum(dim=1), torch.ones(2, 1)))

    def test_adapter_preserves_two_class_contract(self):
        torch.manual_seed(5)
        adapter = TemporalEvidenceAdapter(feature_dim=6, language_dim=5, hidden_dim=8)
        current = torch.randn(2, 3, 6)
        language = torch.randn(2, 5)
        base = torch.randn(2, 3, 2)
        output = adapter(current, language, base_logits=base)
        self.assertEqual(output.shape, base.shape)
        self.assertTrue(torch.isfinite(output).all())
        # The adapter adds an antisymmetric residual and does not normalize over
        # candidates, preserving independent Bernoulli decisions.
        self.assertTrue(torch.allclose((output - base).sum(dim=-1), torch.zeros(2, 3), atol=1e-6))

    def test_sequence_prefix_is_independent_of_future_frames(self):
        torch.manual_seed(6)
        adapter = TemporalEvidenceAdapter(feature_dim=4, language_dim=3, hidden_dim=6)
        adapter.eval()
        current = torch.randn(5, 2, 4)
        bboxes = torch.randn(5, 2, 4)
        track_ids = torch.tensor([10, 11])
        language = torch.randn(2, 3)
        base = torch.randn(5, 2, 2)
        first = run_causal_sequence(
            adapter, current=current, bboxes=bboxes, track_ids=track_ids,
            language=language, namespace="seq/expr", base_logits=base,
        )
        explicit_no_conflict = run_causal_sequence(
            adapter, current=current, bboxes=bboxes, track_ids=track_ids,
            language=language, namespace="seq/expr", base_logits=base,
            candidate_conflict=torch.zeros(5, 2),
        )
        self.assertTrue(torch.allclose(first, explicit_no_conflict, atol=1e-6))
        changed = current.clone()
        changed[3:] = changed[3:] + 100.0
        second = run_causal_sequence(
            adapter, current=changed, bboxes=bboxes, track_ids=track_ids,
            language=language, namespace="seq/expr", base_logits=base,
        )
        self.assertTrue(torch.allclose(first[:3], second[:3], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
