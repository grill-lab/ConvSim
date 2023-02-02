from .AbstractReranker import AbstractReranker
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class CrossEncoderFeedbackRanker(AbstractReranker):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-12-v2').to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-12-v2')
        
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int=100, batch_size=64) -> List[Document]:
        
        if len(conversational_turn.ranking) == 0:
            return conversational_turn.ranking
        
        feedback = conversational_turn.user_utterance
        query = conversational_turn.conversation_history[-1]['rewritten_utterance']
        old_passages = conversational_turn.ranking[max_passages:]
        conversational_turn.ranking = conversational_turn.ranking[:max_passages]

        # passages = [f'{feedback} [SEP] {passage.doc_text}' for passage in conversational_turn.ranking]
        # passages = [passage.doc_text for passage in conversational_turn.ranking]
        passages = [passage.doc_text if passage.doc_text else '' for passage in conversational_turn.ranking]
        # query_list = [query] * len(passages)
        # query_list = [feedback] * len(passages)
        query_list = [f'{query} {feedback}'] * len(passages)
        all_scores = []

        for i in range(0, len(conversational_turn.ranking), batch_size):

            features = self.tokenizer(
                query_list[i:i+batch_size], passages[i:i+batch_size],  
                padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                scores = self.model(**features).logits
                scores = [score[0] for score in scores.tolist()]
                all_scores += scores
        
        for passage, score in zip(conversational_turn.ranking, all_scores):
            passage.score = score
        
        return sorted(conversational_turn.ranking, key=lambda passage: passage.score, reverse=True) + old_passages
