from .AbstractReranker import AbstractReranker
from pygaggle.rerank.base import Query, Text
from .feedback_ranking_utils import FeedbackMonoT5
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List
import re


class T5FeedbackRanker(AbstractReranker):
    def __init__(self):
        self.ranker = FeedbackMonoT5(
            pretrained_model_name_or_path="models/t5-base-msmarco-10k-cast-y4-annotated-feedback-first-1-epochs/checkpoint-2000",
            token_false = '▁false',
            token_true = '▁true'
        )
    
    def rerank(self, conversational_turn: ConversationalTurn, max_passages: int=1000) -> List[Document]:

        if len(conversational_turn.ranking) == 0:
            return conversational_turn.ranking

        feedback = conversational_turn.user_utterance
        search_query = conversational_turn.conversation_history[-1]['rewritten_utterance']
        conversational_turn.ranking = conversational_turn.ranking[:max_passages]

        parsed_query = Query(search_query)
        parsed_passagaes = [
            Text(self.__truncate_passage_text(document.doc_text, limit=10), {'id': document.doc_id}, 0) for document in conversational_turn.ranking
        ][:max_passages]

        reranked_passages = self.ranker.rerank(query=parsed_query, feedback=feedback, texts=parsed_passagaes)
        parsed_passages = []
        for passage in reranked_passages:
            parsed_passage = Document(
                doc_id=passage.metadata['id'],
                doc_text=passage.text,
                score=passage.score
            )
            parsed_passages.append(parsed_passage)
        
        return parsed_passages
    
    def __truncate_passage_text(self, passage_text, limit=5) -> str:
        input_ids = self.ranker.tokenizer.tokenizer(passage_text)
        if len(input_ids) > 512:
            passage_sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', passage_text)
            return ' '.join(passage_sentences[:limit])
        else:
            return passage_text
