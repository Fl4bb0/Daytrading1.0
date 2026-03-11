from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import pandas as pd

from kvant.ml_prepare_data.dataset_preparation_utils import ensure_utc_sorted_index


from typing import Dict, Optional, Protocol
import pandas as pd

class BarSampler(Protocol):
    name: str

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "BarSampler":
        """Tune any sampler parameters using TRAIN ONLY. No-op by default."""
        ...

    def get_global_meta(self) -> dict:
        """Return sampler-level configuration (e.g., target density, grids)."""
        ...

    def get_ticker_meta(self, ticker: str) -> dict:
        """Return per-ticker tuned params (e.g., {"h": 0.02})."""
        ...

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        """Apply sampling to a single ticker series."""
        ...

from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass
class BaseBarSampler:
    """
    Optional convenience base class that implements explicit no-op behavior.
    Subclasses override as needed.
    """
    name: str = "base"

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "BaseBarSampler":
        return self

    def get_global_meta(self) -> dict:
        return {"name": self.name}

    def get_ticker_meta(self, ticker: str) -> dict:
        return {}

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        raise NotImplementedError

@dataclass
class IdentitySampler(BaseBarSampler):
    name: str = "identity"
    subsample_every: int = 1  # optional thinning

    def fit(self, ticker_dfs_train: Dict[str, pd.DataFrame]) -> "IdentitySampler":
        # explicit no-op
        return self

    def get_global_meta(self) -> dict:
        return {
            "name": self.name,
            "subsample_every": int(self.subsample_every),
        }

    def get_ticker_meta(self, ticker: str) -> dict:
        return {}  # no per-ticker params

    def transform(self, df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        df = ensure_utc_sorted_index(df)
        if self.subsample_every > 1:
            df = df.iloc[:: self.subsample_every].copy()
        return df