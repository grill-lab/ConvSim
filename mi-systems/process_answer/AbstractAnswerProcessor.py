import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractAnswerProcessor(ABC):
    def __init__(self):
        """Abstract class for processing answers to CQs.
        """
        pass

    @abstractmethod
    def process_answer(self, clarifying_question: str, answer: str, 
            query: str = None, ranking: List[str] = None) -> str:
        """Process the answer to a clarifying question.

        Args:
            clarifying_question: Clarifying question asked.
            answer: An answer to a clarifying question.
            query: Query string.
            ranking (optional): Ranked list of documents. Defaults to None.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A string of the new query.
        """
        raise NotImplementedError

class DummyAnswerProcessor(AbstractAnswerProcessor):
    def process_answer(self, clarifying_question: str, answer: str, 
            query: str = None, ranking: List[str] = None) -> str:
        """Dummy method that just returns the given answer.
        """
        return answer

