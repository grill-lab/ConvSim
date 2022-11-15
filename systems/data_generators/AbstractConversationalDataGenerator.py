from abc import ABC, abstractmethod
from systems.data_classes import ConversationalTurn


class AbstractConversationalDataGenerator(ABC):
    """
    Abstract Conversational Data Generator Class
    """
    def __init__(self):
        self.topics = None
        self.qrels = None

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
