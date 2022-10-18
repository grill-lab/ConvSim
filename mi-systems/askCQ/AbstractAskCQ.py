import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractAskCQ(ABC):
    def __init__(self):
        """Abstract class for asking clarifying questions.
        """
        pass

    @abstractmethod
    def ask_cq(self, query: str, ranking: List[str] = None) -> str:
        """Ask clarifying question based on query and documents.

        Args:
            query: Query string.
            ranking (optional): Ranked list of documents. Defaults to None.
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
        super().__init__()
            

class GenerateCQ(AbstractAskCQ):
    def __init__(self,):
        """ Abstract class for generating CQs.

        Args:
            tbd.
        """
        pass
            
class DummySelectCQ:
    def __init__(self, question_pool):
        self.question_pool = question_pool
        # super().__init__(question_pool)

    def ask_cq(self, query: str, ranking: List[str] = None) -> str:
        """Dummy method that always returns the first question in pool.
        """
        return self.question_pool[0]

