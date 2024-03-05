from typing import List

from src.data_classes.conversational_turn import ConversationalTurn

from .AbstractFeedbackProvider import AbstractFeedbackProvider


class HumanFeedbackProvider(AbstractFeedbackProvider):

    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
        
        # Display the conversation history
        print("-"*80)
        print("Turn ID: ", conversational_turn.turn_id)
        print(f"Information need: {conversational_turn.information_need}")
        print("===== Conversation history =====")
        for turn in conversational_turn.conversation_history:
            print(f"{turn['participant'].upper()}: {turn['utterance']}")
        
        print(f"USER: {conversational_turn.user_utterance}")
        print(f"SYSTEM: {conversational_turn.system_response}")

        # Prompt the user for feedback
        feedback = input("USER: ")
        return feedback