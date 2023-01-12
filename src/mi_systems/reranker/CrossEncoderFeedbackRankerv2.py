from .CrossEncoderFeedbackRanker import CrossEncoderFeedbackRanker
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List


class CrossEncoderFeedbackRankerv2(CrossEncoderFeedbackRanker):

    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int = 100) -> List[Document]:
        if conversational_turn.feedback_rounds >= 1:
            return super().rerank(conversational_turn, max_passages)
        else:
            return conversational_turn.ranking