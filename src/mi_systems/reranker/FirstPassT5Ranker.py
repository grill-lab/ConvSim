from .T5Ranker import T5Ranker
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List


class FirstPassT5Ranker(T5Ranker):

    def __init__(self):
        super().__init__()
    
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int = 100) -> List[Document]:
        if conversational_turn.feedback_rounds == 0:
            return super().rerank(conversational_turn, max_passages)
        else:
            return conversational_turn.ranking