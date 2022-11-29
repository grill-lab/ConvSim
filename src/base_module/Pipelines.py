from .AbstractModule import AbstractModule
from src.simulator.provide_feedback import AbstractFeedbackProvider

from typing import List
from src.data_classes.conversational_turn import ConversationalTurn


class Pipeline(AbstractModule):
    def __init__(self, modules: List[AbstractModule]) -> None:
        self.modules = modules

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:
            conversational_turn = module.step(conversational_turn)

        return conversational_turn


class RecursivePipeline(Pipeline):

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:
            conversational_turn = module.step(conversational_turn)
            if isinstance(module, AbstractFeedbackProvider) and \
                conversational_turn.feedback_rounds < 3 and \
                    conversational_turn.evaluate_turn() < 0.5:
                conversational_turn.feedback_rounds += 1
                conversational_turn = self.step(conversational_turn)

        return conversational_turn
