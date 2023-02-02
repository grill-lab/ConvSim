from .T5Rewriter import T5Rewriter
from src.data_classes.conversational_turn import ConversationalTurn


class DiscourseAwareFeedbackRewriter(T5Rewriter):

    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        if conversational_turn.feedback_rounds > 0:
            precision_at_3 = conversational_turn.evaluate_turn(measure='P(rel=2)@3')
            precision_at_1 = conversational_turn.evaluate_turn(measure='P(rel=2)@1')

            if precision_at_3 > 0.3 or precision_at_1 == 1.0:
                return conversational_turn.conversation_history[-1]['rewritten_utterance']
            else:
                return super().rewrite(conversational_turn)