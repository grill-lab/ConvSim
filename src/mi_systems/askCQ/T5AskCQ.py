from src.data_classes.conversational_turn import ConversationalTurn
from transformers import pipeline

from .AbstractAskCQ import AbstractAskCQ


class T5AskCQ(AbstractAskCQ):

    def __init__(self, model_path: str = "../data/models/t5-base-clarify"):
        self.clarifier = pipeline(
            "text2text-generation", 
            model=model_path, 
            tokenizer=model_path, 
            device_map="auto"
        )
    
    def ask_cq(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        parsed_conversation = self.__parse_conversation(conversational_turn)
        # Generate the rewrite.
        candidates = self.clarifier(
            parsed_conversation, max_length=256)
        return candidates[0]['generated_text']

    def __parse_conversation(self, conversational_turn: ConversationalTurn) -> str:
        """Format the conversations for inference."""
        # Format the conversations.
        conversation = ""
        for turn in conversational_turn.conversation_history:
            if turn['participant'] == 'User':
                conversation += " ||| " + turn['utterance']
            if turn['participant'] == 'System':
                conversation += " ||| " + turn['utterance']
        conversation += " ||| " + conversational_turn.user_utterance
        conversation = "Clarify: " + conversation
        return conversation