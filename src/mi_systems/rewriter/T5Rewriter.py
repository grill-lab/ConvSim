from src.data_classes.conversational_turn import ConversationalTurn
from transformers import pipeline

from .AbstractRewriter import AbstractRewriter


class T5Rewriter(AbstractRewriter):
    def __init__(self, model_path: str = "castorini/t5-base-canard"):
        self.rewriter = pipeline(
            "text2text-generation", 
            model=model_path, 
            tokenizer=model_path, 
            device_map="auto"
        )
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        context = self.__parse_conversation(conversational_turn)
        rewrite = self.rewriter(
            context, max_length=256, repetition_penalty=2.5,
            length_penalty=1.0, early_stopping=True)[0]['generated_text']

        return rewrite
    
    def __parse_conversation(self, conversational_turn: ConversationalTurn) -> str:
        """Format the conversations for inference."""
        # Format the conversations.
        previous_utterances = [turn['utterance'] for turn in conversational_turn.conversation_history]
        previous_utterances += [conversational_turn.user_utterance]
        context = " ||| ".join(previous_utterances)
        return context