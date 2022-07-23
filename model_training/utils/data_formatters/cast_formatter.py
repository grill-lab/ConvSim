from .abstract_data_formatter import AbstractDataFormatter
import pandas as pd
import json

class CAsTFormatter(AbstractDataFormatter):

    def convert_to_dataframe(self, data_path: str) -> pd.DataFrame:
        
        dataset = []

        with open(data_path) as topics_file:
            topics = json.load(topics_file)
            for topic in topics:
                description = topic['description']
                for index, turn in enumerate(topic['turns']):
                    context = [f"{previous_turn['utterance']} ||| {previous_turn['system_response']}" for previous_turn in topic['turns'][:index]]
                    context = " ||| ".join(context)
                    utterance = turn['utterance']

                    dataset.append({
                        'source_text' : f"Topic: {description} ||| Conversation: {context}",
                        'target_text' : utterance
                    })
        
        dataset = pd.DataFrame(dataset)
        # dataset = dataset.sample(frac=1).reset_index(drop=True)
        
        return dataset