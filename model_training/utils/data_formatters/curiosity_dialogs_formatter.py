from .abstract_data_formatter import AbstractDataFormatter

from datasets import load_dataset, DatasetDict
import json
import urllib.request

class CuriosityDialogsFormatter(AbstractDataFormatter):

    def __format_samples(self, sample: dict) -> dict:
        sample['context'] = f"Topic: {sample['topic']} Facets: {','.join(sample['facets'])} Knowledge: {','.join(sample['prior_knowledge'])} History: {sample['context']}"
        return sample

    def convert_to_jsonlines(self, data_path: str) -> None:

        with urllib.request.urlopen(data_path) as url:
            dialogs = json.loads(url.read().decode())
        
        data = []

        for dialog in dialogs['dialogs']:
            facets = dialog['aspects']
            topic = dialog['focus_entity']
            prior_knowledge = dialog['known_entities']
            
            for idx, utterance in enumerate(dialog['messages']):
                if utterance['sender'] == 'user':
                    target = utterance['message']
                    context = " ||| ".join([message['message'] for message in dialog['messages'][:idx]])

                    data.append({
                        "context": context,
                        "target" : target,
                        "facets" : facets,
                        "topic" : topic,
                        "prior_knowledge": prior_knowledge
                    })
        
        with open("/shared/training_data/curiosity_dialogs_data.jsonl", "w") as data_file:
            for item in data:
                json.dump(item, data_file)
                data_file.write('\n')
    
    
    def load_dataset(self, data_path: str) -> DatasetDict:

        dataset = load_dataset('json', data_files=data_path)
        dataset = dataset.map(self.__format_samples)
        dataset = dataset.remove_columns(['facets', 'topic', 'prior_knowledge'])
        dataset = dataset['train'].train_test_split(test_size=0.3)

        return dataset
