from abc import abstractmethod
from typing import List

from data_classes.conversational_turn import ConversationalTurn, Document
from src.base_module.AbstractModule import AbstractModule


class AbstractReranker(AbstractModule):
    def __init__(self):
        """Abstract class for reranking."""
        pass

    @abstractmethod
    def rerank(self, conversational_turn: ConversationalTurn) -> List[Document]:
        """Method for initial retrieval.

        Args:
            conversational_turn: A class representing conversational turn,
                containing query, history, clarfifying question, answer.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            Re-ranked list of documents.
        """
        raise NotImplementedError

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        ranking = self.rerank(conversational_turn)
        # TODO: should we have an utterance here or not? Maybe top1 passage?
        return conversational_turn.update_history(
            "", participant="System", utterance_type="ranking", ranking=ranking
        )
