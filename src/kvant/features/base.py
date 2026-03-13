"""
features.base — Abstract base for all feature engineers.

Concrete implementations:
  - OHLCVFeatures          (ohlcv.py)         — raw price/volume columns
  - IntradayTA10Features   (ta.py)            — 10 intraday technical indicators
  - StandardizedFeatures   (standardized.py)  — wraps any base FE with z-score scaling
  - ArticleEmbeddingFeatures (nlp.py)         — LLM/BERT article embeddings (future)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class FeatureEngineer(ABC):
    """
    Sklearn-style fit/transform interface over OHLCV DataFrames.

    fit()       — estimate any statistics (e.g. mean/std) from SAMPLED TRAIN data.
    transform() — return (X, feature_names) for any split.
    """

    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit scaling statistics or other parameters on sampled training data."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Return:
          X            — float32 array of shape (n_bars, n_features)
          feature_names — list of column name strings, length n_features
        """
        ...
