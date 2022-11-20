from abc import abstractmethod

from src.base_module.AbstractModule import AbstractModule
from src.data_classes.conversational_turn import ConversationalTurn


class AbstractFeedbackProvider(AbstractModule):
    def __init__(self) -> None:
        """Abstract class for answering clarifying questions."""
        pass

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        response = self.give_feedback(conversational_turn)

        conversational_turn.update_history(response, "User", utterance_type="feedback")
        return conversational_turn

    @abstractmethod
    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
        """Answers gives feedback to a system response based on information need.

        Args:
            conversational_turn: Object containing all information about a turn.
        """
        raise NotImplementedError
