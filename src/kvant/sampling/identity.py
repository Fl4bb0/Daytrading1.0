"""
sampling.identity — Pass-through sampler with optional uniform thinning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from kvant.sampling.base import BarSampler
from kvant.utils.time_utils import ensure_utc_sorted_index


@dataclass
class IdentitySampler(BarSampler):
    """
    Returns every bar unchanged, with an optional ``subsample_every`` stride.

    Parameters
    ----------
    subsample_every : int
        Keep every N-th row (default 1 = keep all).
    """
    name: str = "identity"
    subsample_every: int = 1

    def fit(self, ticker_dfs: Dict[str, pd.DataFrame]) -> "IdentitySampler":
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name, "subsample_every": int(self.subsample_every)}

    def transform(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if self.subsample_every > 1:
            df = df.iloc[:: self.subsample_every].copy()
        return df
