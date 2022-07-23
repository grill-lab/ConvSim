from simplet5 import SimpleT5
from config import dataset_config
from sklearn.model_selection import train_test_split
from utils.data_formatters.abstract_data_formatter import AbstractDataFormatter

from IPython import embed
import pandas as pd
import re

class QulacFormatter(AbstractDataFormatter):

    def convert_to_dataframe(self, data_path: str) -> pd.DataFrame:
        dataset = []
        df = pd.read_json(data_path)
        for i, row in df.iterrows():
            description = re.sub(r"[^a-zA-Z0-9'? ]", '', row['facet_desc'])
            question = re.sub(r"[^a-zA-Z0-9'? ]", '', row['question'])
            answer = re.sub(r"[^a-zA-Z0-9'? ]", '', row['answer'])
            dataset.append({
                'source_text' : f"Topic: {description} ||| Conversation: {question}",
                'target_text' : answer
            })
        return pd.DataFrame(dataset)


dataset_config = [{
        "name" : "qulac",
        "data_formatter" : QulacFormatter,
        "source" : "/home/sekulic/proj/qulac/data/qulac/qulac.json" # TODO: change
        }]

if __name__ == "__main__":

    for data in dataset_config:

        data_formatter = data["data_formatter"]()
        data_source = data["source"]

        dataset = data_formatter.convert_to_dataframe(data_source)
        train_df, test_df = train_test_split(dataset, test_size=0.10)

        model = SimpleT5()
        model.from_pretrained(model_type="t5", model_name="t5-base")
        model.train(
            train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=512, 
            target_max_token_len=128, 
            batch_size=2, 
            max_epochs=5, 
            use_gpu=True,
            outputdir=f"models/AnswerCQ-finetuned-{data['name']}"
        )


