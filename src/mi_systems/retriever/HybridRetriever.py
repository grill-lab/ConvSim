from typing import List

from src.data_classes.conversational_turn import ConversationalTurn, Document

from .AbstractRetriever import AbstractRetriever
from .DenseRetriever import DenseRetriever
from .SparseRetriever import SparseRetriever


class HybridRetriever(AbstractRetriever):

    def __init__(self, 
                 dense_searcher: DenseRetriever, 
                 sparse_searcher: SparseRetriever):
        super().__init__(None)
        self.dense_searcher = dense_searcher
        self.sparse_searcher = sparse_searcher
    
    def retrieve(self, conversational_turn: ConversationalTurn, 
                 num_results: int = 1000) -> List[Document]:
        
        dense_query = " ".join(
            [turn["utterance"] for turn in conversational_turn.conversation_history])
        sparse_query = conversational_turn.rewritten_utterance if \
            conversational_turn.rewritten_utterance else conversational_turn.user_utterance
        
        dense_results = self.dense_searcher.retrieve(
            dense_query, num_results)
        sparse_results = self.sparse_searcher.retrieve(
            sparse_query, num_results)
        
        return self._merge_results(dense_results, sparse_results)
        
    def _merge_results(self,
                        dense_results: List[Document],
                        sparse_results: List[Document],
                        alpha: float = 0.1, normalization: bool = False,
                        weight_on_dense: bool = False
                        ) -> List[Document]:
        
        merged_results = []

        dense_ranking_lookup = {
            document.doc_id: document for document in dense_results}
        sparse_ranking_lookup = {
            document.doc_id: document for document in sparse_results}

        dense_scores = [document.score for document in dense_results]
        sparse_scores = [document.score for document in sparse_results]

        min_dense_score = min(dense_scores) if dense_scores else 0
        max_dense_score = max(dense_scores) if dense_scores else 0
        min_sparse_score = min(sparse_scores) if sparse_scores else 0
        max_sparse_score = max(sparse_scores) if sparse_scores else 0

        for doc_id in set(dense_ranking_lookup.keys()) | set(sparse_ranking_lookup.keys()):
            if doc_id not in dense_ranking_lookup:
                document = sparse_ranking_lookup[doc_id]
                sparse_score = document.score
                dense_score = min_dense_score
            elif doc_id not in sparse_ranking_lookup:
                document = dense_ranking_lookup[doc_id]
                dense_score = document.score
                sparse_score = min_sparse_score
            else:
                dense_score = dense_ranking_lookup[doc_id].score
                sparse_score = sparse_ranking_lookup[doc_id].score

            if normalization:
                sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2) \
                    / (max_sparse_score - min_sparse_score)
                dense_score = (dense_score - (min_dense_score + max_dense_score) / 2) \
                    / (max_dense_score - min_dense_score)

            score = alpha * sparse_score + \
                dense_score if not weight_on_dense else sparse_score + alpha * dense_score
            document.score = score
            merged_results.append(document)

        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results
