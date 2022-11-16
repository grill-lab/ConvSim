from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from data_classes.conversational_turn import ConversationalTurn, Document
from src.base_module.AbstractModule import AbstractModule


class AbstractRetriever(AbstractModule):
    def __init__(self, collection):
        """Abstract class for initial retrieval.

        Args:
            collection: Document collection.
        """
        self.collection = collection

    # rethink if pd.DataFrame is the best way to store this
    @abstractmethod
    def retrieve(
        self, conversational_turn: ConversationalTurn, num_results: int = 1000
    ) -> List[Document]:
        """Method for initial retrieval.

        Args:
            conversational_turn: A class representing the conversational turn.
            num_results: Number of docs to return (defaults to 1000).
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            Ranked list of documents.
        """
        raise NotImplementedError

    def batch_retrieve(
        self, conversational_turns: List[ConversationalTurn], num_results: int = 1000
    ) -> List[List[Document]]:
        """Batch retrieval based on self.retrieve.

        Args:
            conversational_turns: A list of ConversationalTurn.
            num_results: Number of docs to return (defaults to 1000).
        Returns:
            List of list of ranked documents.
        """
        return [self.retrieve(ct, num_results) for ct in conversational_turns]

    def step(
        self, conversational_turn: ConversationalTurn, num_results: int = 1000
    ) -> ConversationalTurn:
        ranking = self.retrieve(conversational_turn, num_results=num_results)
        # TODO: should we have an utterance here or not? Maybe top1 passage?
        return conversational_turn.update_history(
            "", participant="System", utterance_type="ranking", ranking=ranking
        )


class DummyRetriever(AbstractRetriever):
    def __init__(self, collection):
        super().__init__(collection)

    def retrieve(
        self, conversational_turn: ConversationalTurn, num_results: int = 1000
    ) -> pd.DataFrame:
        """Dummy method that just returns top num_results from collection."""
        return self.collection[:num_results]
