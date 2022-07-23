from .abstract_data_formatter import AbstractDataFormatter

import pandas as pd
import json
import urllib.request

class CuriosityDialogsFormatter(AbstractDataFormatter):

    def convert_to_dataframe(self, data_path: str) -> pd.DataFrame:

        with urllib.request.urlopen(data_path) as url:
            dialogs = json.loads(url.read().decode())
        
        dataset = []

        for dialog in dialogs['dialogs']:
            facets = f"Facets: {dialog['aspects']}"
            topic = f"Topic: {dialog['focus_entity']}"
            prior_knowledge = f"Knowledge: {dialog['known_entities']}"
            
            for idx, utterance in enumerate(dialog['messages']):
                if utterance['sender'] == 'user':
                    target = utterance['message']
                    context = " ||| ".join([message['message'] for message in dialog['messages'][:idx]])

                    dataset.append({
                        "source_text": topic + " ||| " + facets + " ||| " + prior_knowledge + " ||| " + "Conversation: " + context,
                        "target_text" : target
                    })
        
        dataset = pd.DataFrame(dataset)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        
        return dataset