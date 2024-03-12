from src.data_classes.conversational_turn import ConversationalTurn
from transformers import pipeline

from .AbstractResponseGenerator import AbstractRespnseGenerator


class T5ResponseGenerator(AbstractRespnseGenerator):

    def __init__(self,
                 model_path: str = "../../data/models/t5-response-generator/"
                 ) -> None:
        self.summariser = pipeline(
            "text2text-generation", 
            model=model_path, 
            tokenizer=model_path, 
            device_map="auto"
        )
    
    def generate_response(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        parsed_passages = "Summarize: " + self.__parse_passages(
            conversational_turn)
        # Generate the rewrite.
        return self.summariser(
            parsed_passages, max_length=512)[0]['generated_text']

    def __parse_passages(self, conversational_turn: ConversationalTurn) -> str:
        """Format the passages for inference."""
        # Format the conversations.
        return " ".join(
            f"({idx}): {passage}" for idx, passage in enumerate(
                conversational_turn.ranking[:3], start=1
            )
        )