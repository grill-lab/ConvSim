from abc import ABC, abstractmethod
from ..data_classes import ConversationalTurn


class AbstractConversationalDataGenerator(ABC):
    """
    Abstract Conversational Data Generator Class
    """
    def __init__(self, dataset_path: str, relevance_judgements_path: str):
        self.dataset_path = dataset_path
        self.relevance_judgements_path = relevance_judgements_path

    @abstractmethod
    def get_turn(self) -> ConversationalTurn:
        """Yields a Conversational Turn.

        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A Conversational Turn with information need description, user
            utterance, conversation history, and turn specific relevance
            judgements.
        """
        raise NotImplementedError
