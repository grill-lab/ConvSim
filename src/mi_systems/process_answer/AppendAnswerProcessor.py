from src.mi_systems.process_answer.AbstractAnswerProcessor import AbstractAnswerProcessor

from src.data_classes.conversational_turn import ConversationalTurn


class AppendAnswerProcessor(AbstractAnswerProcessor):
    def __init__(self):
        """Qulac-style processor where CQ and answer are appended to query."""
        super().__init__()

    def process_answer(self, conversational_turn: ConversationalTurn) -> str:
        """Appends a clarifying question and an answer to the user query."""
        # TODO: check if we get query, CQ, and answer this way
        rewrite = " ".join(
            [
                conversational_turn.conversation_history[-1][
                    "rewritten_utterance"
                ],  # query
                conversational_turn.system_response,  # CQ
                conversational_turn.user_utterance,  # answer
            ]
        )

        return rewrite
