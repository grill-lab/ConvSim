# pip install rank_bm25
import pandas as pd
from rank_bm25 import BM25Okapi

from src.data_classes.conversational_turn import ConversationalTurn
from src.mi_systems.askCQ.AbstractAskCQ import SelectCQ


class BM25AskCQ(SelectCQ):
    def __init__(
        self, question_pool_path: str, model_name: str = "all-mpnet-base-v2"
    ) -> None:
        """Semantic matching based clarifying question selection class."""
        self.question_pool = self.load_question_pool(question_pool_path)
        self.corpus = self.question_pool.dropna().question.values
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def load_question_pool(self, question_pool_path: str) -> pd.DataFrame:
        return pd.read_json(question_pool_path)

    def ask_cq(self, conversational_turn: ConversationalTurn) -> str:
        # TODO: think about whether it's user_utterance or rewritten_utterance
        tokenized_query = conversational_turn.user_utterance.split(" ")
        cq = self.bm25.get_top_n(tokenized_query, self.corpus, n=1)[0]
        return cq
