from abc import ABC, abstractmethod

from data_classes.conversational_turn import ConversationalTurn
from data_classes.ranking import Ranking


class AbstractAnswerProcessor(ABC):
    def __init__(self):
        """Abstract class for processing answers to CQs."""
        pass

    @abstractmethod
    def process_answer(
        self, conversational_turn: ConversationalTurn, ranking: Ranking = None
    ) -> str:
        """Process the answer to a clarifying question.

        Args:
            conversational_turn: A class representing conversational turn,
                containing query, history, clarfifying question, answer.
            ranking (optional): Class representing ranking. Defaults to None.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A string of the new query. #TODO: think about this.
        """
        raise NotImplementedError


class DummyAnswerProcessor(AbstractAnswerProcessor):
    def process_answer(self, conversational_turn: ConversationalTurn) -> str:
        """Dummy method that just returns the given answer."""
        return conversational_turn.answer
