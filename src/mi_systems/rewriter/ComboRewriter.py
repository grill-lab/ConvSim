from src.data_classes.conversational_turn import ConversationalTurn
from transformers import pipeline

from .AbstractRewriter import AbstractRewriter
from .T5FeedbackRewriter_v2 import T5FeedbackRewriterv2
from .T5Rewriter import T5Rewriter


class ComboRewriter(AbstractRewriter):

    def __init__(self):
        self.main_rewriter = T5Rewriter()
        self.feedback_rewriter = T5FeedbackRewriterv2()
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        if conversational_turn.user_utterance_type == "Feedback":
            return self.feedback_rewriter.rewrite(conversational_turn)
        else:
            return self.main_rewriter.rewrite(conversational_turn)