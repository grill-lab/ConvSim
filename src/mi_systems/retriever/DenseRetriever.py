import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List

from bs4 import BeautifulSoup as bs
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher
from src.data_classes.conversational_turn import ConversationalTurn, Document

from .AbstractRetriever import AbstractRetriever


class DenseRetriever(AbstractRetriever):

    def __init__(self, dense_shards, collection, 
                 encoder='castorini/tct_colbert-v2-hnp-msmarco', 
                 collection_type="trecweb"):
        super().__init__(collection)
        self.doc_retriever = LuceneSearcher(self.collection)
        self.dense_shards = dense_shards
        self.encoder_str = encoder
        self.collection_type = collection_type
    
    def retrieve(self, conversational_turn: ConversationalTurn, 
                 num_results: int = 1000, max_workers=2) -> List[Document]:
        # query = conversational_turn.rewritten_utterance if \
        #     conversational_turn.rewritten_utterance else conversational_turn.user_utterance
        query = " ".join(
            [turn["utterance"] for turn in conversational_turn.conversation_history])
        query += " " + conversational_turn.user_utterance
        
        def __retrieve(args):
            shard_idx, shard = args
            gpu_id = shard_idx % max_workers
            # Update encoder.
            encoder = TctColBertQueryEncoder(self.encoder_str, 
                                             device="cuda")
            searcher = FaissSearcher(shard, encoder)
            search_results = searcher.search(
                query, num_results // len(self.dense_shards), 4
            )
            parsed_passages = self._parse_search_results(search_results)

            del searcher
            return parsed_passages
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            parsed_passages = executor.map(__retrieve, enumerate(self.dense_shards))
        # parsed_passages = [
        #     __retrieve(args) for args in enumerate(self.dense_shards)]
        
        parsed_passages = [
            passage for passage_list in parsed_passages for passage in passage_list]
        # sort passages by score
        parsed_passages.sort(key=lambda x: x.score, reverse=True)
        return parsed_passages[:num_results]
    
    def _parse_search_results(self, search_results) -> List[Document]:
        passages = []        
        if self.collection_type == "trecweb":
            for result in search_results:
                doc_id, passage_id = result.docid.split(":")
                document = self.doc_retriever.doc(doc_id).raw()
                document = bs(document, "lxml")
                sub_passages = document.find_all("passage")
                for idx, passage in enumerate(sub_passages):
                    if idx == int(passage_id):
                        passages.append(Document(
                            doc_id=result.docid,
                            doc_text=passage.text,
                            score=result.score
                        ))
                        break
        
        return passages
