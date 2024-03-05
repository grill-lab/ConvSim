from .AbstractReranker import AbstractReranker
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List


class T5Ranker(AbstractReranker):
    def __init__(self):
        self.ranker = MonoT5()
    
    def rerank(self, conversational_turn: ConversationalTurn, 
               max_passages: int=1000) -> List[Document]:

        if len(conversational_turn.ranking) == 0:
            return conversational_turn.ranking

        search_query = conversational_turn.rewritten_utterance if \
            conversational_turn.rewritten_utterance else conversational_turn.user_utterance
        parsed_query = Query(search_query)
        parsed_passagaes = [
            Text(document.doc_text, {'id': document.doc_id}, 0) for document in conversational_turn.ranking
        ][:max_passages]

        reranked_passages = self.ranker.rerank(parsed_query, parsed_passagaes)
        parsed_passages = []
        for passage in reranked_passages:
            parsed_passage = Document(
                doc_id=passage.metadata['id'],
                doc_text=passage.text,
                score=passage.score
            )
            parsed_passages.append(parsed_passage)
        
        return parsed_passages
        