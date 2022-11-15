from .AbstractConversationalDataGenerator import (
    AbstractConversationalDataGenerator
)
import json
from ir_measures import read_trec_qrels
from src.data_classes import ConversationalTurn


class CAsTY4DataGenerator(AbstractConversationalDataGenerator):

    def __init__(self):
        dataset_path = \
            'data/cast/year_4/2022_evaluation_topics_flattened_duplicated_v1.0.json'
        relevance_judgements_path = 'data/cast/year_4/cast2022.qrel'
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
                information_need = turn.get('information_need')
                utterance = turn.get("utterance")
                relevance_judgements = [
                    qrel for qrel in self.qrels if qrel.query_id == turn_id]
                conversational_history = []
                for previous_turns in topic['turn'][:index]:
                    previous_user_utterance = previous_turns.get("utterance")
                    previous_system_response = previous_turns.get("response")
                    conversational_history += [
                        {"User": previous_user_utterance}, 
                        {"System": previous_system_response}
                    ]
                yield ConversationalTurn(
                    turn_id=turn_id, information_need=information_need,
                    user_utterance=utterance,
                    conversation_history=conversational_history,
                    relevance_judgements=relevance_judgements
                )
