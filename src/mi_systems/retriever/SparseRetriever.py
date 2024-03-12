import json
from typing import List

from bs4 import BeautifulSoup as bs
from pyserini.search.lucene import LuceneSearcher
from src.data_classes.conversational_turn import ConversationalTurn, Document

from .AbstractRetriever import AbstractRetriever


class SparseRetriever(AbstractRetriever):

    def __init__(self, collection, collection_type="json"):
        super().__init__(collection)
        self.retriever = LuceneSearcher(self.collection)
        self.retriever.set_bm25(4.46, 0.82)
        self.collection_type = collection_type

    def retrieve(self, conversational_turn: ConversationalTurn,
                 num_results: int = 1000) -> List[Document]:
        search_query = conversational_turn.rewritten_utterance if \
            conversational_turn.rewritten_utterance else \
            conversational_turn.user_utterance

        search_results = self.retriever.search(search_query, num_results)
        parsed_passages = self._parse_search_results(search_results)
        return parsed_passages

    def _parse_search_results(self, search_results, sep=":") -> List[Document]:
        parsed_passages = []
        if self.collection_type == "json":
            for search_result in search_results:
                parsed_search_result = json.loads(search_result.raw)
                parsed_passage = Document(
                    doc_id=search_result.docid,
                    doc_text=parsed_search_result["contents"],
                    score=search_result.score
                )
                parsed_passages.append(parsed_passage)

        return parsed_passages
