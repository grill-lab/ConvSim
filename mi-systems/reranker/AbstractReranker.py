import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractReranker(ABC):
    def __init__(self):
        """Abstract class for reranking.
        """
        pass

	# rethink if pd.DataFrame is the best way to store this
    @abstractmethod
    def rerank(self, query: str, ranking: pd.DataFrame) -> pd.DataFrame:
        """Method for initial retrieval.

        Args:
            query: Query string.
            ranking: pandas DataFrame of initial ranking.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            Re-ranked list of documents in a trec-like format.
        """
        raise NotImplementedError

