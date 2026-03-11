"""
experiment.config — Experiment identity and configuration dataclass.

ExperimentConfig is the single source of truth for an experiment run.
stable_id() produces a short SHA-256 digest so every unique config maps
to a unique output directory automatically.
"""
import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Fully describes a data-preparation + training experiment.

    Fields
    ------
    experiment_name   : human-readable label (does NOT affect stable_id).
    sampler           : serialised BarSampler config dict  (from dataclasses.asdict).
    feature_engineer  : serialised FeatureEngineer config dict.
    labeler           : serialised Labeler config dict.
    lookback_L        : lookback window length (number of bars) fed into the model.
    """
    experiment_name: str
    sampler: dict
    feature_engineer: dict
    labeler: dict
    lookback_L: int

    def stable_id(self) -> str:
        """16-char hex digest that uniquely identifies this config."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]
