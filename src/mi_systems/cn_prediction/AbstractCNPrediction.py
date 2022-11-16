from abc import ABC, abstractmethod

from data_classes.conversational_turn import ConversationalTurn


# This module shouldn't inherit AbstractModule perhaps.
# The ConversationalTurn doesn't need updating
# It can be a part of AskCQ class.
class AbstractCNPrediction(ABC):
    def __init__(self):
        """Abstract class for predicting clarification need."""
        pass

    @abstractmethod
    def predict_cn(self, conversational_turn: ConversationalTurn) -> bool:
        """Predict if asking clarifying question is needed or not.

        Args:
            conversational_turn: A class representing conversational turn.
            ranking: A class representing a ranking.
        """
        raise NotImplementedError


class DummyCNPrediction(AbstractCNPrediction):
    def predict_cn(self, conversational_turn: ConversationalTurn) -> bool:
        """Always return True (i.e., clarifying question is needed)."""
        return True
