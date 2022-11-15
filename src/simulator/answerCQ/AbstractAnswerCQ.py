from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class AbstractAnswerCQ(ABC):
    def __init__(self) -> None:
        """Abstract class for answering clarifying questions."""
        pass

    @abstractmethod
    def answer_cq(self, clarifying_question: str, query: str = None) -> str:
        """Answers given clarifying question based on self.information_need.

        Args:
            clarifying_question: A question to answer.
            query: Associated query that can be used as additional context.
        """
        raise NotImplementedError
