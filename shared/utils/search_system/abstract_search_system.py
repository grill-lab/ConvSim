from abc import ABC, abstractmethod
from typing import List


class AbstractSearchSystem(ABC):

    @abstractmethod
    def rewrite_query(self, query: str, context: List) -> str:
        """
        Rewrite a query given query and context (conversation history)

        Args:
            query: Input query to be rewritten, string
            context: Conversation history with all turns (user and system) 
            in the conversation so far. Format is [u1, s1, u2, s2, ...]

        Returns:
            A query rewrite useful for downstream retrieval
        """
        pass

    @abstractmethod
    def retrieve_documents(self, query: str) -> List:
        """
        Retrieves documents given an input (usually rewritten) query

        Args:
            query: Input query for retrieval

        Returns:
            A list of documents determined relevant to the input query
        """
        pass

    @abstractmethod
    def rank_passages(self, query: str, documents: List) -> List:
        """
        Ranks extracted passages output of the retrieve_documents methods 
        according to their relevance to the input query

        Args:
            query: Input query for retrieval
            documents: List of documents retrieved from search system. Passages 
            need to be extracted from documents first before ranking

        Returns:
            A list passages ranked according to their relevance to input query
        """
        pass

    @abstractmethod
    def generate_summary(self, passages: List, threshold: int) -> str:
        """
        Generates a summary from the top-n passages ranked by generate_summary
        method.

        Args:
            passages: Ranked list of passages. Top-n used to generate summary
            treshold: The number of top passages to generate summary from

        Returns:
            A summary of the top-n passages. This constitutes the output
            of the search system
        """
        pass
