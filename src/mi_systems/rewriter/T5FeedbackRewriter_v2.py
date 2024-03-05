from src.data_classes.conversational_turn import ConversationalTurn
from transformers import pipeline

from .AbstractRewriter import AbstractRewriter


class T5FeedbackRewriterv2(AbstractRewriter):

    def __init__(self, model_path: str = "../data/models/tuned-t5-base-rewriter-v1-2e3-20epochs/"):
        self.rewriter = pipeline(
            "text2text-generation", 
            model=model_path, 
            tokenizer=model_path, 
            device_map="auto"
        )

    
    def rewrite(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        parsed_conversation = self.__parse_conversation(conversational_turn)
        rewrite = self.rewriter(
                parsed_conversation, max_length=64)[0]['generated_text']
        
        return rewrite

    def __parse_conversation(self, conversational_turn: ConversationalTurn) -> str:
        """Format the conversations for inference."""
        # Format the conversations.
        previous_utterances = [turn['utterance'] for turn in conversational_turn.conversation_history]
        previous_utterances += [conversational_turn.user_utterance]
        context = " </s> ".join(previous_utterances)
        context = "reformulate: " + context
        return context