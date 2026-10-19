"""Causal temporal evidence components for the Refer-Sports FlexHook study.

The package is deliberately independent of the upstream FlexHook checkout.  It
can therefore be unit-tested on CPU and injected into a controlled evaluator
without changing the frozen baseline source tree.
"""

from .evidence_buffer import EvidenceBuffer, EvidenceSnapshot
from .reliability_gate import ReliabilityGate, ReliabilityWeights
from .temporal_referring import FLEXHOOK_VARIANTS, TemporalReferringHead
from .adapter import CausalSequenceOutput, TemporalEvidenceAdapter, run_causal_sequence

__all__ = [
    "EvidenceBuffer",
    "EvidenceSnapshot",
    "ReliabilityGate",
    "ReliabilityWeights",
    "TemporalReferringHead",
    "FLEXHOOK_VARIANTS",
    "CausalSequenceOutput",
    "TemporalEvidenceAdapter",
    "run_causal_sequence",
]
