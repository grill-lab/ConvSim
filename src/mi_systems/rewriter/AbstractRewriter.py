from abc import abstractmethod

from src.data_classes.conversational_turn import ConversationalTurn
from src.base_module.AbstractModule import AbstractModule


class AbstractRewriter(AbstractModule):
    def __init__(self):
        """Abstract class for rewriting."""
        pass

    @abstractmethod
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        """Method for query rewriting.

        Args:
            conversational_turn: A class representing conversational turn,
                containing query, history, clarfifying question, answer.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            Rewritten user utterance.
        """
        raise NotImplementedError

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        rewrite = self.rewrite(conversational_turn)
        if rewrite:
            conversational_turn.rewritten_utterance = rewrite
        return conversational_turn
