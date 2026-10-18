"""Research-only FGRefMOT-v2 components.

The modules in this package are deliberately usable without changing the
original ``temp_rmot`` graph.  This keeps the published/baseline path
available while making each new component testable in isolation.
"""

from .text_evidence import TextEvidence, TextEvidenceEncoder
from .track_evidence_sampler import TrackEvidenceSampler
from .referring_pair_head import (
    ReferringMLPHead,
    ReferringPairHead,
    build_parameter_matched_controls,
)
from .tam_v2 import TAMV2Output, TrackCentricTAMV2
from .temporal_evidence import TemporalEvidenceEncoder, TemporalEvidenceOutput
from .track_memory import ReliabilityAwareTrackMemory, TrackMemoryState
from .qim_writer import EvidenceQueryWriter, QueryWriterOutput
from .qim_v2 import (
    QIMV2Output,
    ReliabilityAwareQIMV2,
    ReliabilityAwareQueryInteraction,
)
from .training_losses import (
    FGRefV2LossWeights,
    compose_fgref_v2_losses,
    identity_consistency_loss,
    masked_binary_referring_loss,
)

__all__ = [
    "TextEvidence",
    "TextEvidenceEncoder",
    "TrackEvidenceSampler",
    "ReferringMLPHead",
    "ReferringPairHead",
    "build_parameter_matched_controls",
    "TAMV2Output",
    "TrackCentricTAMV2",
    "TemporalEvidenceEncoder",
    "TemporalEvidenceOutput",
    "ReliabilityAwareTrackMemory",
    "TrackMemoryState",
    "EvidenceQueryWriter",
    "QueryWriterOutput",
    "QIMV2Output",
    "ReliabilityAwareQIMV2",
    "ReliabilityAwareQueryInteraction",
    "FGRefV2LossWeights",
    "compose_fgref_v2_losses",
    "identity_consistency_loss",
    "masked_binary_referring_loss",
]
