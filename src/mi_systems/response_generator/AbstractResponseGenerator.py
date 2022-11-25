from abc import abstractmethod

from src.data_classes.conversational_turn import ConversationalTurn
from src.base_module.AbstractModule import AbstractModule


class AbstractRespnseGenerator(AbstractModule):
    def __init__(self):
        """Abstract class for reranking."""
        pass

    @abstractmethod
    def generate_response(self, conversational_turn: ConversationalTurn) -> str:
        """Generates a response from the top N documents in ranking.

        Args:
            conversational_turn: A class representing conversational turn,
                containing query, history, clarfifying question, answer.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A generated response.
        """
        raise NotImplementedError

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        response = self.generate_response(conversational_turn)
        conversational_turn.update_history(
            utterance=response,
            participant="System",
            utterance_type="response"
        )
        return conversational_turn