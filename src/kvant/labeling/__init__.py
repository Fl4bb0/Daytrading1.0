# labeling — label generators (supervised targets)

# labeling — bar labeling methods
from kvant.labeling.base import Labeler
from kvant.labeling.triple_barrier import (
    TripleBarLabel,
    TripleBarrierLabeler,
    triple_barrier_label,
)

# Registry: name → class, used by experiment configs and sweeps
LABELER_REGISTRY: dict[str, type[Labeler]] = {
    "triple_barrier": TripleBarrierLabeler,
}

__all__ = [
    "Labeler",
    "TripleBarLabel",
    "TripleBarrierLabeler",
    "triple_barrier_label",
    "LABELER_REGISTRY",
]
