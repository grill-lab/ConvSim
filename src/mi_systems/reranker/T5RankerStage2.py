from .T5Ranker import T5Ranker
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List


class T5RankerStage2(T5Ranker):
    
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int = 100) -> List[Document]:
        # if len(conversational_turn.ranking) > 0:
        #     old_passges = conversational_turn.ranking[max_passages:]
        #     return super().rerank(conversational_turn, max_passages) + old_passges
        # else:
        #     return conversational_turn.ranking
        old_passges = conversational_turn.ranking[max_passages:]
        return super().rerank(conversational_turn, max_passages) + old_passges
