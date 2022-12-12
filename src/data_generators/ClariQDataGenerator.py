import pandas as pd
from ir_measures import read_trec_qrels

from src.data_classes import ConversationalTurn

from .AbstractConversationalDataGenerator import AbstractConversationalDataGenerator


class ClariQDataGenerator(AbstractConversationalDataGenerator):
    def __init__(self,  dataset_path: str):
        self.data = pd.read_csv(dataset_path, sep='\t')
        
        # TODO: add qrels

    def get_turn(self) -> ConversationalTurn:
        """The function will iterate over all rows in ClariQ.
        TODO: add an option to iterate only over unique facet descriptions;
        TODO: add qrels.
        """
        for i, row in self.data.iterrows():
            turn_id = row["facet_id"]+"-"+row["question_id"]
            information_need = row["facet_desc"]
            utterance = row["initial_request"]
            utterance_type = "question"
            relevance_judgements = []

            turn = ConversationalTurn(
                turn_id=turn_id,
                information_need=information_need,
                user_utterance=utterance,
                user_utterance_type=utterance_type,
                relevance_judgements=relevance_judgements
            )

            turn.update_history(
                utterance=row["question"],
                participant="System",
                utterance_type="clarifying_question")

            yield turn
