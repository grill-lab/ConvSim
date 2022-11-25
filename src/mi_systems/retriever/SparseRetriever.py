from .AbstractRetriever import AbstractRetriever
from pyserini.search.lucene import LuceneSearcher

from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List
import json


class SparseRetriever(AbstractRetriever):

    def __init__(self, collection, collection_type="json"):
        super().__init__(collection)
        self.retriever = LuceneSearcher(self.collection)
        self.retriever.set_bm25(4.46, 0.82)
        self.collection_type = collection_type
    
    def retrieve(self, conversational_turn: ConversationalTurn, 
        num_results: int = 1000) -> List[Document]:
        search_query = conversational_turn.rewritten_utterance if \
            conversational_turn.rewritten_utterance else conversational_turn.user_utterance
        
        search_results = self.retriever.search(search_query, num_results)
        parsed_passages = self._parse_search_results(search_results)
        return parsed_passages
    
    def _parse_search_results(self, search_results) -> List[Document]:
        parsed_passages = []
        if self.collection_type == "json":
            for search_result in search_results:
                parsed_search_result = json.loads(search_result.raw)
                passages = parsed_search_result["passage_splits"]
                for passage in passages:
                    parsed_passage = Document(
                        doc_id= f"{search_result.docid}-{passage['id']}",
                        doc_text=passage["body"],
                        score=search_result.score
                    )
                    parsed_passages.append(parsed_passage)
        
        return parsed_passages

