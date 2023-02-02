from .T5Rewriter import T5Rewriter
from src.data_classes.conversational_turn import ConversationalTurn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class T5FeedbackRewriter(T5Rewriter):
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        if conversational_turn.feedback_rounds > 0:
            return super().rewrite(conversational_turn)

