# inspired by iai-group

import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractRetriever(ABC):
    def __init__(self, collection):
        """Abstract class for initial retrieval.

        Args:
            collection: Document collection.
        """
        self.collection = collection

    # rethink if pd.DataFrame is the best way to store this
    @abstractmethod
    def retrieve(self, query: str, num_results: int = 1000) -> pd.DataFrame:
        """Method for initial retrieval.

        Args:
            query: Query string.
            num_results: Number of docs to return (defaults
                to 1000).
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            Ranked list of documents in a trec-like format.
        """
        raise NotImplementedError

    def batch_retrieve(self, queries: List[str]) -> List[pd.DataFrame]:
        """Batch retrieval based on self.retrieve.

        Args:
            queries: List of input queries.
        Returns:
            List of rankings.
        """
        return [self.retrieve(query) for query in queries]

class DummyRetriever(AbstractRetriever):
    def retrieve(self, query: str, num_results: int = 1000) -> pd.DataFrame:
        """Dummy method that just returns top num_results from collection.
        """
        return self.collection[:num_results]

