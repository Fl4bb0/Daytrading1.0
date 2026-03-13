"""
labeling.base — Base class for all labelers.

Concrete implementations:
  - TripleBarrierLabeler  (triple_barrier.py)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class Labeler:
    """
    Convenience base class with explicit no-op defaults.

    fit()       — estimate any data-driven parameters from training data. No-op by default.
    transform() — return (y, metadata) where:
                    y        — int8 array of shape (n_bars,)
                               convention: -1 = abstain / no label
                    metadata — list of per-bar dicts (or None), same length as y
    """

    name: str = "base"

    def fit(self, df: pd.DataFrame) -> "Labeler":
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[Optional[dict]]]:
        raise NotImplementedError(f"{self.__class__.__name__}.transform() not implemented.")
