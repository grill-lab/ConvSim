from abc import ABC, abstractmethod
from typing import List

from data_classes.conversational_turn import ConversationalTurn


class AbstractAskCQ(ABC):
    def __init__(self):
        """Abstract class for asking clarifying questions."""
        pass

    @abstractmethod
    def ask_cq(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        """Ask clarifying question based on query, ranked list of docs etc.

        Args:
            conversational_turn: A class representing conversational turn.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A ConversationalTurn with a clarifying_question attribute updated.
        """
        raise NotImplementedError


class SelectCQ(AbstractAskCQ):
    def __init__(self, question_pool):
        """Abstract class for selecting CQ from predefined pool of questions.

        Args:
            question_pool: Path to a predefined pool of questions.
        """
        self.question_pool = question_pool
        super().__init__()


class GenerateCQ(AbstractAskCQ):
    def __init__(
        self,
    ):
        """Abstract class for generating CQs.

        Args:
            tbd.
        """
        pass


class DummySelectCQ:
    def __init__(self, question_pool):
        self.question_pool = question_pool
        # super().__init__(question_pool)

    def ask_cq(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        """Dummy method that always returns the first question in pool."""
        # TODO: what else needs to happen? update conversation history?

        question = self.question_pool[0]
        self.update_history(
            system_response=question, system_response_type="clarifying_question"
        )

        return conversational_turn
