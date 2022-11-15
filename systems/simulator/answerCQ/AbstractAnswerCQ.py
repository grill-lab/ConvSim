from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class AbstractAnswerCQ(ABC):
    def __init__(self) -> None:
        """Abstract class for answering clarifying questions."""
        pass

    def set_information_need(self, information_need: str) -> None:
        """Set information need that the answers will be based upon.

        Args:
            information_need: Textual description of the information need to
            base the answers on.
        """
        self.information_need = information_need

    @abstractmethod
    def answer_cq(self, clarifying_question: str, query: str = None) -> str:
        """Answers given clarifying question based on self.information_need.

        Args:
            clarifying_question: A question to answer.
            query: Associated query that can be used as additional context.
        """
        raise NotImplementedError
