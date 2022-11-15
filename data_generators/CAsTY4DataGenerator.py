from .AbstractConversationalDataGenerator import (
    AbstractConversationalDataGenerator
)
import json
from ir_measures import read_trec_qrels
from ..data_classes import ConversationalTurn


class CAsTY4DataGenerator(AbstractConversationalDataGenerator):

    def __init__(self, dataset_path: str, relevance_judgements_path: str):
        super().__init__(dataset_path, relevance_judgements_path)
        
        with open(dataset_path) as cast_y4_topics_file:
            self.topics = json.load(cast_y4_topics_file)
        
        self.qrels = list(read_trec_qrels(relevance_judgements_path))
    
    def get_turn(self) -> ConversationalTurn:
        parsed_turns = set()
        for topic in self.topics:
            for index, turn in enumerate(topic['turn']):
                turn_id = f"{topic['number']}_{turn['number']}"
                if turn_id in parsed_turns:
                    continue
                information_need = turn.get('information_need', 'Sample Information need')
                conversational_history = []
                for previous_turns in turns[:index]:
                    pass