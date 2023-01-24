from pygaggle.rerank.transformer import MonoT5
from pygaggle.model.tokenize import QueryDocumentBatchTokenizer, QueryDocumentBatch, QueryDocumentBatchTokenizer
from pygaggle.rerank.base import Query, Text
from pygaggle.model import greedy_decode
from transformers import AutoTokenizer
from typing import List, Iterable
from dataclasses import dataclass
from copy import deepcopy
import torch


@dataclass
class QueryDocumentFeedbackBatch(QueryDocumentBatch):
    feedback: str = ""
    def __init__(self, query, documents, feedback, output=None) -> None:
        super().__init__(query=query, documents=documents, output=output)
        self.feedback = feedback


class QueryFeedbackDocumentBatchTokenizer(QueryDocumentBatchTokenizer):
    def traverse_query_feedback_document(self, batch_input: QueryDocumentFeedbackBatch) -> Iterable[QueryDocumentFeedbackBatch]:
        query = batch_input.query
        feedback = batch_input.feedback
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.documents[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                                        query=query.text,
                                        feedback=feedback,
                                        document=doc.text) for doc in docs])
            yield QueryDocumentFeedbackBatch(query, docs, feedback, outputs)


class FeedbackT5BatchTokenizer(QueryFeedbackDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Feedback: {feedback} Document: {document} Relevant:'
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512
        super().__init__(*args, **kwargs)


class FeedbackMonoT5(MonoT5):

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str, *args, batch_size: int = 8, **kwargs) -> FeedbackT5BatchTokenizer:
        return FeedbackT5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs), batch_size=batch_size
        )
    
    def rerank(self, query: Query, feedback: str, texts: List[Text]) -> List[Text]:
        """Sorts a list of texts
        """
        return sorted(self.rescore(query, feedback, texts), key=lambda x: x.score, reverse=True)

    
    def rescore(self, query: Query, feedback:str, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentFeedbackBatch(query=query, feedback=feedback, documents=texts)
        for batch in self.tokenizer.traverse_query_feedback_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score

        return texts