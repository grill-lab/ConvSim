from .T5Rewriter import T5Rewriter
from src.data_classes.conversational_turn import ConversationalTurn


class T5FeedbackRewriter(T5Rewriter):

    def __init__(self):
        super().__init__()
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        if conversational_turn.feedback_rounds > 0:
            return super().rewrite(conversational_turn)
        else:
            return None
