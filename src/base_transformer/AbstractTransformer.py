from abc import ABC, abstractmethod
from src.data_classes.conversational_turn import ConversationalTurn
from typing import List


class AbstractTransformer(ABC):

    @abstractmethod
    def step(self, conversational_turn: ConversationalTurn) \
            -> ConversationalTurn:
        pass


class Pipeline(AbstractTransformer):

    def __init__(self, transformers: List[AbstractTransformer]) -> None:
        self.transformers = transformers

    def step(self, conversational_turn: ConversationalTurn) \
            -> ConversationalTurn:
        for transformer in self.transformers:
            conversational_turn = transformer.step(conversational_turn)

        return conversational_turn
