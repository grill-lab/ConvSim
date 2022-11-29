from abc import ABC, abstractmethod

from src.base_module.AbstractModule import AbstractModule
from src.data_classes.conversational_turn import ConversationalTurn


class AbstractAskCQ(AbstractModule):
    def __init__(self):
        """Abstract class for asking clarifying questions."""
        pass

    @abstractmethod
    def ask_cq(self, conversational_turn: ConversationalTurn) -> str:
        """Ask clarifying question based on query, ranked list of docs etc.

        Args:
            conversational_turn: A class representing conversational turn.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A string representing the clarifying question.
        """
        raise NotImplementedError

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        question = self.ask_cq(conversational_turn)
        conversational_turn.update_history(
            question, participant="System", utterance_type="clarifying_question"
        )
        return conversational_turn


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


# TODO: should this inherit from SelectCQ or not?
class DummySelectCQ(SelectCQ):
    def __init__(self, question_pool):
        super().__init__(question_pool)

    def ask_cq(self, conversational_turn: ConversationalTurn) -> str:
        """Dummy method that always returns the first question in pool."""
        # we just provide the question, .step updates history and all
        question = self.question_pool[0]
        return question
