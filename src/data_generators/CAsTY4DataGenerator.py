from .AbstractConversationalDataGenerator import (
    AbstractConversationalDataGenerator
)
import json
from ir_measures import read_trec_qrels
from src.data_classes import ConversationalTurn


class CAsTY4DataGenerator(AbstractConversationalDataGenerator):

    def __init__(self, dataset_path: str, relevance_judgements_path: str):
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
                information_need = turn.get("information_need")
                utterance = turn.get("utterance")
                manual_utterance = turn.get("manual_rewritten_utterance")
                utterance_type = turn.get("utterance_type").lower()
                relevance_judgements = [
                    qrel for qrel in self.qrels if qrel.query_id == turn_id]
                conversational_history = []
                for previous_turn in topic['turn'][:index]:
                    # extract utterance attributes
                    previous_user_utterance = previous_turn.get("utterance")
                    previous_user_utterance_rewrite = previous_turn.get(
                        "automatic_rewritten_utterance")
                    previous_user_utterance_type = previous_turn.get(
                        "utterance_type")

                    # extract response attributes
                    previous_system_response = previous_turn.get("response")
                    previous_system_response_type = previous_turn.get(
                        "response_type")

                    conversational_history += [
                        {
                            "participant": "User", 
                            "utterance": previous_user_utterance,
                            "utterance_type": previous_user_utterance_type, 
                            "rewritten_utterance": previous_user_utterance_rewrite
                        },
                        {
                            "participant": "System", 
                            "utterance": previous_system_response,
                            "utterance_type": previous_system_response_type
                        }
                    ]
                parsed_turns.add(turn_id)
                yield ConversationalTurn(
                    turn_id=turn_id, information_need=information_need,
                    user_utterance=utterance,
                    manual_utterance=manual_utterance,
                    conversation_history=conversational_history,
                    user_utterance_type=utterance_type,
                    relevance_judgements=relevance_judgements
                )
