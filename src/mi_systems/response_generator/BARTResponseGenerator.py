from .AbstractResponseGenerator import AbstractRespnseGenerator
from transformers import pipeline
import torch

from src.data_classes.conversational_turn import ConversationalTurn

class BARTResponseGenerator(AbstractRespnseGenerator):
    
    def __init__(self):
        self.summariser = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn", 
            device=-1 if not torch.cuda.is_available() else 0
        )
    
    def generate_response(self, conversational_turn: ConversationalTurn, k=3) -> str:
        top_passages = '\n'.join([document.doc_text for document in conversational_turn.ranking[:k]])
        output = self.summariser(
            top_passages, 
            max_length=200, 
            min_length=30, 
            do_sample=False,
            truncation=True
        )
        return output[0]['summary_text']
