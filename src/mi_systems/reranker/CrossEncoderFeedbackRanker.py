from .AbstractReranker import AbstractReranker
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class CrossEncoderFeedbackRanker(AbstractReranker):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'models/feedback-cross-encoder-2023-01-06_16-09-53').to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'models/feedback-cross-encoder-2023-01-06_16-09-53')
        
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int=100) -> List[Document]:
        
        if len(conversational_turn.ranking) == 0:
            return conversational_turn.ranking
        
        feedback = conversational_turn.user_utterance
        query = conversational_turn.conversation_history[-1]['utterance']
        conversational_turn.ranking = conversational_turn.ranking[:max_passages]

        passages = [f'{feedback} [SEP] {passage.doc_text}' for passage in conversational_turn.ranking]
        query_list = [query] * len(passages)

        features = self.tokenizer(
            query_list, passages,  padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits
            scores = [score[0] for score in scores.tolist()]
        
        for passage, score in zip(conversational_turn.ranking, scores):
            passage.score = score
        
        return sorted(conversational_turn.ranking, key=lambda passage: passage.score, reverse=True)
