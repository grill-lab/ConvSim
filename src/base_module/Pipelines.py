from .AbstractModule import AbstractModule
from src.simulator.provide_feedback import AbstractFeedbackProvider

from typing import List
from src.data_classes.conversational_turn import ConversationalTurn


class Pipeline(AbstractModule):
    """Single pass through all conversational modules"""

    def __init__(self, modules: List[AbstractModule]) -> None:
        self.modules = modules

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:
            conversational_turn = module.step(conversational_turn)

        return conversational_turn


class RecursivePipeline(Pipeline):
    """Allows for multiple feedback rounds"""

    def __init__(self, modules: List[AbstractModule], max_feedback_rounds=3, min_measure=1.0) -> None:
        super().__init__(modules)
        self.max_feedback_rounds = max_feedback_rounds
        self.min_measure = min_measure

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:
            conversational_turn = module.step(conversational_turn)
            if isinstance(module, AbstractFeedbackProvider) and \
                conversational_turn.feedback_rounds < self.max_feedback_rounds and \
                    conversational_turn.evaluate_turn() < self.min_measure:
                conversational_turn.feedback_rounds += 1
                conversational_turn = self.step(conversational_turn)

        return conversational_turn
