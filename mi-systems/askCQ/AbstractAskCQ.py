import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractAskCQ(ABC):
    def __init__(self):
        """Abstract class for asking clarifying questions.
        """
        pass

    @abstractmethod
    def ask_cq(self, query: str, ranking: List[str]) -> str:
        """Ask clarifying question based on query and documents.

        Args:
            query: Query string.
            ranking: Ranked list of documents.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A string of clarifiying question to ask.
        """
        raise NotImplementedError


class SelectCQ(AbstractAskCQ):
    def __init__(self, question_pool):
        """ Abstract class for selecting CQ from predefined pool of questions.

        Args:
            question_pool: Path to a predefined pool of questions.
        """
        self.question_pool = question_pool
            

class GenerateCQ(AbstractAskCQ):
    def __init__(self,):
        """ Abstract class for generating CQs.

        Args:
            tbd.
        """
        pass
            
