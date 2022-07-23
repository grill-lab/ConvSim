
from .abstract_data_formatter import AbstractDataFormatter
import pandas as pd
import json
from IPython import embed

class QulacFormatter(AbstractDataFormatter):

    def convert_to_dataframe(self, data_path: str) -> pd.DataFrame:
        dataset = []
        for row in pd.read_json(data_path):
            embed()

            dataset.append({
                'source_text' : f"Topic: {description} ||| Conversation: {context}",
                'target_text' : utterance
            })
            embed()
        return pd.DataFrame(dataset)


