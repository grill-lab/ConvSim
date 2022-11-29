from abc import abstractmethod

from src.data_classes.conversational_turn import ConversationalTurn
from src.base_module.AbstractModule import AbstractModule


class AbstractAnswerProcessor(AbstractModule):
    def __init__(self):
        """Abstract class for processing answers to CQs."""
        pass

    @abstractmethod
    def process_answer(self, conversational_turn: ConversationalTurn) -> str:
        """Process the answer to a clarifying question.

        Args:
            conversational_turn: A class representing conversational turn,
                containing query, history, clarfifying question, answer.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A string of the new query. #TODO: think about this.
        """
        raise NotImplementedError

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        """Process the answer to CQ and update ConvTurn."""
        utterance = self.process_answer(conversational_turn)
        conversational_turn.rewritten_utterance = utterance
        return conversational_turn


class DummyAnswerProcessor(AbstractAnswerProcessor):
    def process_answer(self, conversational_turn: ConversationalTurn) -> str:
        """Dummy method that just returns the given answer."""
        return conversational_turn.answer
