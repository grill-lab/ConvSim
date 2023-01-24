from .T5Rewriter import T5Rewriter
from src.data_classes.conversational_turn import ConversationalTurn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class T5FeedbackRewriter(T5Rewriter):

    def __init__(self):
        # super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            'models/t5_base_canard-all-samples-lr-1e-3-1-epoch/checkpoint-1000'
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'models/t5_base_canard-all-samples-lr-1e-3-1-epoch/checkpoint-1000', truncation_side='left'
        )
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        if conversational_turn.feedback_rounds > 0:
            return super().rewrite(conversational_turn)

