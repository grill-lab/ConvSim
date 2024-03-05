from .AbstractResponseGenerator import AbstractRespnseGenerator
from transformers import pipeline
import torch

from src.data_classes.conversational_turn import ConversationalTurn

class BARTResponseGenerator(AbstractRespnseGenerator):
    
    def __init__(self):
        self.summariser = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn", 
            # change depending on GPU availability
            device=-1 if not torch.cuda.is_available() else 2
            # device=-1 if not torch.cuda.is_available() else 3
            # device_map="auto"
        )
    
    def generate_response(self, conversational_turn: ConversationalTurn, k=3) -> str:
        top_passages = '\n'.join([document.doc_text for document in conversational_turn.ranking[:k]])
        output = self.summariser(
            top_passages, 
            max_length=256, 
            min_length=32, 
            do_sample=False,
            truncation=True
        )
        return output[0]['summary_text']
