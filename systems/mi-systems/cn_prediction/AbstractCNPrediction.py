from abc import ABC, abstractmethod

from data_classes.conversational_turn import ConversationalTurn
from data_classes.ranking import Ranking


class AbstractCNPrediction(ABC):
    def __init__(self):
        """Abstract class for predicting clarification need."""
        pass

    @abstractmethod
    def predict_cn(
        self, conversational_turn: ConversationalTurn, ranking: Ranking = None
    ) -> bool:
        """Predict if asking clarifying question is needed or not.

        Args:
            conversational_turn: A class representing conversational turn.
            ranking: A class representing a ranking.
        """
        raise NotImplementedError


class DummyCNPrediction(AbstractCNPrediction):
    def predict_cn(
        self, conversational_turn: ConversationalTurn, ranking: Ranking = None
    ) -> bool:
        """Always return True (i.e., clarifying question is needed."""
        return True
