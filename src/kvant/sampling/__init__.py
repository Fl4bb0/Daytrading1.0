# sampling — bar sampling / event-driven sub-sampling
from kvant.sampling.base import BarSampler
from kvant.sampling.identity import IdentitySampler
from kvant.sampling.cusum import TunedCUSUMBarSampler
from kvant.sampling.count import TunedTickBarSampler, TunedVolumeBarSampler, TunedDollarBarSampler
from kvant.sampling.imbalance import TunedTIBSampler, TunedVIBSampler, TunedDIBSampler

# Registry: name → class, used by experiment configs and sweeps
SAMPLER_REGISTRY: dict[str, type[BarSampler]] = {
    "identity": IdentitySampler,
    "cusum":    TunedCUSUMBarSampler,
    "tick":     TunedTickBarSampler,
    "volume":   TunedVolumeBarSampler,
    "dollar":   TunedDollarBarSampler,
    "tib":      TunedTIBSampler,
    "vib":      TunedVIBSampler,
    "dib":      TunedDIBSampler,
}

__all__ = [
    "BarSampler",
    "IdentitySampler",
    "TunedCUSUMBarSampler",
    "TunedTickBarSampler",
    "TunedVolumeBarSampler",
    "TunedDollarBarSampler",
    "TunedTIBSampler",
    "TunedVIBSampler",
    "TunedDIBSampler",
    "SAMPLER_REGISTRY",
]
