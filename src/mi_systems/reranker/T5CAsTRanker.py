from .AbstractReranker import AbstractReranker
from src.data_classes.conversational_turn import ConversationalTurn, Document
from transformers import AutoTokenizer
from .BaseT5PassageRanker import BaseT5PassageRanker
from typing import List
import torch


class T5CAsTRanker(AbstractReranker):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained('castorini/monot5-base-msmarco-10k')
        self.ranker = BaseT5PassageRanker.from_pretrained(
            'castorini/monot5-base-msmarco-10k').to(self.device).eval()
    
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int=100, batch_size: int=32) -> List[Document]:
        if len(conversational_turn.ranking) == 0:
            return conversational_turn.ranking
        
        feedback = conversational_turn.user_utterance
        query = conversational_turn.conversation_history[-1]['utterance']
        conversational_turn.ranking = conversational_turn.ranking[:max_passages]

        for i in range(0, len(conversational_turn.ranking), 64):
            parsed_passages = [
                f'Query: {query} Feedback: {feedback} Document: {document.doc_text}' for \
                    document in conversational_turn.ranking][i:i+batch_size]
        
            with torch.no_grad():
                model_inputs = self.tokenizer(parsed_passages, max_length=512, truncation=True, padding=True, return_tensors='pt')
                scores = self.ranker(**model_inputs)
            
            for document, score in zip(conversational_turn.ranking[i:i+batch_size], scores):
                document.score = score
        
        return sorted(conversational_turn.ranking, key=lambda item: item.score, reverse=True)
