from abc import ABC, abstractmethod
from typing import List

from src.data_classes.conversational_turn import ConversationalTurn


class AbstractModule(ABC):
    def __init__(self):
        """Abstract class for all modules of mixed-initiative systems."""
        pass

    def __call__(self, *input, **kwargs):
        return self.step(*input, **kwargs)

    @abstractmethod
    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        """Each mi-system and simulator module needs to modify this method.

        Args:
            conversational_turn: A class representing conversational turn. They
                might differ from module to module.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            A ConversationalTurn with updated attributes.
        """
        raise NotImplementedError
