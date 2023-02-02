from .SparseRetriever import SparseRetriever
from src.data_classes.conversational_turn import ConversationalTurn, Document
from typing import List, Dict
from bs4 import BeautifulSoup as bs


class RunFileRetriever(SparseRetriever):

    def __init__(self, collection, collection_type="json", run_file=None):
        super().__init__(collection, collection_type)
        self.results: Dict[str, List[Document]] = self._parse_run_file(run_file)

    def retrieve(self, conversational_turn: ConversationalTurn, num_results=1000) -> List[Document]:
        turn_id = conversational_turn.turn_id
        documents: List[Document] = self.results[turn_id]
        current_score: float = 0.0
        for document in documents:
            document.doc_text = self._get_passage_text(document.doc_id)
            document.score = current_score
            current_score -= 1.0
        return documents

    
    def _parse_run_file(self, run_file) -> Dict[str, List[Document]]:
        parsed_documents: Dict[str, List[Document]] = dict()
        with open(run_file) as f:
            for line in f:
                turn_id, _, passage_id, _, score, _ = line.split()
                if turn_id not in parsed_documents:
                    parsed_documents[turn_id] = [
                        Document(doc_id=passage_id, doc_text=None, score=float(score))
                    ]
                else:
                    parsed_documents[turn_id].append(
                        Document(doc_id=passage_id, doc_text=None, score=float(score))
                    )
            return parsed_documents

    
    def _get_passage_text(self, passage_id):
        doc_id, p_index = passage_id.rsplit('-', 1)[0], passage_id.rsplit('-', 1)[1]
        doc = self.retriever.doc(doc_id)
        doc = bs(doc.raw(), "lxml")
        passages = doc.find_all("passage")
        text = [passage.text for passage in passages if str(passage['id'])==p_index]
        if text:
            return text[0]