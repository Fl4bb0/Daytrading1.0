"""
signals.base — Abstract base for all text/LLM-based signal extractors.

A SignalExtractor converts a set of raw articles into a numeric signal
Series aligned to a price DatetimeIndex. The signal can then be consumed
by a FeatureEngineer (features/nlp.py) or a fusion model.

Concrete implementations:
  - SentimentSignal  (sentiment.py)  — FinBERT / VADER sentiment score
  - LLMSignal        (llm.py)        — GPT/Claude article summarise → bullish score
  - EmbeddingSignal  (embedding.py)  — raw CLS-token embedding (feeds into fusion model)
"""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class SignalExtractor(ABC):
    """
    Convert raw article records into a float signal aligned to a price index.

    extract() is the single required method. It receives a list of article dicts
    (each with at minimum 'published_utc', 'ticker', 'headline', 'body') and a
    target DatetimeIndex, and returns a float Series of the same length.

    NaN should be used where no signal is available for a given bar.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def extract(
        self,
        articles: List[dict],
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Parameters
        ----------
        articles : list of dicts, each containing at least
                   'published_utc' (pd.Timestamp, UTC),
                   'ticker'        (str),
                   'headline'      (str),
                   'body'          (str).
        index    : target DatetimeIndex (UTC-naive, sorted ascending).

        Returns
        -------
        pd.Series of float, same length as index, NaN where no signal.
        """
        ...
