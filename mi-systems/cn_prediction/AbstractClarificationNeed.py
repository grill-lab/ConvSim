import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractCNPrediction(ABC):
    def __init__(self):
        """Abstract class for predicting clarification need.
        """
        pass

    @abstractmethod
    def predict_cn(self, query: str, ranking: List[str] = None) -> bool:
        """Predict if asking clarifying question is needed or not.
        
        Args:
            query: Query string.
            ranking: Ranked list of documents.
        """
        raise NotImplementedError


