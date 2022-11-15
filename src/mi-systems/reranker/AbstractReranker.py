from abc import ABC, abstractmethod

from data_classes.conversational_turn import ConversationalTurn
from data_classes.ranking import Ranking


class AbstractReranker(ABC):
    def __init__(self):
        """Abstract class for reranking."""
        pass

    @abstractmethod
    def rerank(
        self, conversational_turn: ConversationalTurn, ranking: Ranking
    ) -> Ranking:
        """Method for initial retrieval.

        Args:
            conversational_turn: A class representing conversational turn,
                containing query, history, clarfifying question, answer.
            ranking: Class representing ranking.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            Re-ranked list of documents in Ranking class.
        """
        raise NotImplementedError
