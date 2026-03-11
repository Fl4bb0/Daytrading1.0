"""
kdata.news — Abstract base for news / article data retrievers.

Concrete implementations will live alongside this file, e.g.:
  - RSSNewsRetriever
  - NewsAPIRetriever
  - FinnhubNewsRetriever
"""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class NewsRetriever(ABC):
    """Fetch news articles for a list of ticker symbols."""

    @abstractmethod
    def get_articles(
        self,
        symbols: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with at minimum the columns:
          - published_utc  (datetime, UTC-naive)
          - ticker         (str)
          - headline       (str)
          - body           (str, may be empty)
          - source         (str)
        """
        ...
