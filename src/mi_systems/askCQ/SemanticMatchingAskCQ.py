import pandas as pd
from sentence_transformers import SentenceTransformer, util

from src.data_classes.conversational_turn import ConversationalTurn
from src.mi_systems.askCQ.AbstractAskCQ import SelectCQ


class SemanticMatchingAskCQ(SelectCQ):
    def __init__(
        self, question_pool_path: str, model_name: str = "all-mpnet-base-v2"
    ) -> None:
        """Semantic matching based clarifying question selection class."""
        self.question_pool = self.load_question_pool(question_pool_path)
        self.embedder = SentenceTransformer(model_name)
        self.embeddings = self.embed_question_pool()

    def load_question_pool(self, question_pool_path: str) -> pd.DataFrame:
        return pd.read_json(question_pool_path)

    def embed_question_pool(self) -> None:
        """Embed question pool with embedder."""
        return self.embedder.encode(
            self.question_pool.question.values, convert_to_tensor=True
        )

    def ask_cq(self, conversational_turn: ConversationalTurn) -> str:
        query_embedding = self.embedder.encode(
            conversational_turn.user_utterance, convert_to_tensor=True
        )
        top1 = util.semantic_search(query_embedding, self.embeddings, top_k=1)
        cq = self.question_pool.iloc[top1[0][0]["corpus_id"]].question
        return cq
