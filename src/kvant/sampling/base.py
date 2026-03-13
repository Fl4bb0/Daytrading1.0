"""
sampling.base — Abstract base for all bar samplers.

Concrete implementations:
  - IdentitySampler        (identity.py)   — pass-through / uniform thinning
  - TunedCUSUMBarSampler   (cusum.py)      — CUSUM event-driven filter
  - TunedTickBarSampler    (count.py)      — tick bars
  - TunedVolumeBarSampler  (count.py)      — volume bars
  - TunedDollarBarSampler  (count.py)      — dollar bars
  - TunedTIBSampler        (imbalance.py)  — tick imbalance bars
  - TunedVIBSampler        (imbalance.py)  — volume imbalance bars
  - TunedDIBSampler        (imbalance.py)  — dollar imbalance bars
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class BarSampler:
    """
    Convenience base class with explicit no-op defaults.

    fit()            — estimate per-ticker tuning params from TRAIN only.
    transform()      — return the sub-sampled DataFrame for a single ticker.
    get_global_meta() — serialisable global metadata dict.
    get_ticker_meta() — serialisable per-ticker metadata dict.
    """

    name: str = "base"

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "BarSampler":
        """Estimate tuning parameters from training data only. No-op by default."""
        return self

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__}.transform() not implemented.")

    def get_global_meta(self) -> dict:
        return {"name": self.name}

    def get_ticker_meta(self, ticker: str) -> Optional[dict]:
        return None
