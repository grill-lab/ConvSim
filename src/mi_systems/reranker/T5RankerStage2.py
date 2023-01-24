from .T5Ranker import T5Ranker
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List


class T5RankerStage2(T5Ranker):

    def __init__(self):
        self.ranker = MonoT5(
            pretrained_model_name_or_path="models/t5-base-msmarco-10k-cast-y4-annotated-feedback-first-1-epochs-concat-query+feedback/checkpoint-2000",
            token_false = '▁false',
            token_true = '▁true'
        )
    
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int = 1000) -> List[Document]:
        return super().rerank(conversational_turn, max_passages)