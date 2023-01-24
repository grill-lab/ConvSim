from .AbstractReranker import AbstractReranker
from src.data_classes.conversational_turn import ConversationalTurn, Document
from transformers import AutoTokenizer
from .BaseT5PassageRanker import BaseT5PassageRanker
from typing import List
import torch


class T5CAsTRanker(AbstractReranker):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained('models/t5-base-msmarco-10k-cast-y4-annotated-feedback-first-1-epochs/checkpoint-2000')
        self.ranker = BaseT5PassageRanker.from_pretrained(
            'models/t5-base-msmarco-10k-cast-y4-annotated-feedback-first-1-epochs/checkpoint-2000').to(self.device).eval()
    
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int=1000, batch_size: int=64) -> List[Document]:
        if len(conversational_turn.ranking) == 0:
            return conversational_turn.ranking
        
        feedback = conversational_turn.user_utterance
        query = conversational_turn.conversation_history[-1]['rewritten_utterance']
        conversational_turn.ranking = conversational_turn.ranking[:max_passages]

        for i in range(0, len(conversational_turn.ranking), batch_size):
            parsed_passages = [
                f'Query: {query} Feedback: {feedback} Document: {document.doc_text} Relevant:' for \
                    document in conversational_turn.ranking][i:i+batch_size]
        
            with torch.no_grad():
                model_inputs = self.tokenizer(parsed_passages, max_length=512, truncation=True, padding=True, return_tensors='pt')
                scores = self.ranker(**model_inputs)
            
            for document, score in zip(conversational_turn.ranking[i:i+batch_size], scores):
                document.score = score
        
        return sorted(conversational_turn.ranking, key=lambda item: item.score, reverse=True)
