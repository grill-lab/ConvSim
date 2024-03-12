from .AbstractConversationalDataGenerator import (
    AbstractConversationalDataGenerator
)
import json
from ir_measures import read_trec_qrels
from src.data_classes import ConversationalTurn
from typing import Generator


class IKATDataGenerator(AbstractConversationalDataGenerator):

    def __init__(self, dataset_path: str, relevance_judgements_path: str) :
        with open(dataset_path) as ikat_topics_file:
            self.topics = json.load(ikat_topics_file)

        self.qrels = list(read_trec_qrels(relevance_judgements_path))
    
    def get_turn(self) -> Generator[ConversationalTurn, None, None]:
        parsed_turns = set()
        for topic in self.topics:
            ptkbs = topic.get("ptkb")
            for index, turn in enumerate(topic['turns']):
                turn_id = f"{topic['number']}_{turn['turn_id']}"
                if turn_id in parsed_turns:
                    continue
                information_need = turn.get("information_need")
                utterance = turn.get("utterance")
                utterance_type = turn.get("utterance_type")
                relevance_judgements = [
                    qrel for qrel in self.qrels if qrel.query_id == turn_id]
                ptkb_provenance = turn.get("ptkb_provenance")
                if ptkb_provenance:
                    user_preferences = [ptkbs.get(str(ptkb_idx)) for ptkb_idx in ptkb_provenance]
                else:
                    user_preferences = ["I do not have any preferences."]

                conversational_history = []
                for previous_turn in topic['turns'][:index]:
                    previous_user_utterance = previous_turn.get("utterance")
                    previous_user_utterance_type = previous_turn.get(
                        "utterance_type")
                    previous_user_resolved_utterance = previous_turn.get(
                        "resolved_utterance")
                    
                    previous_system_response = previous_turn.get("response")
                    previous_system_response_type = previous_turn.get(
                        "response_type")

                    conversational_history += [
                        {
                            "participant": "User", 
                            "utterance": previous_user_utterance,
                            "utterance_type": previous_user_utterance_type,
                            "resolved_utterance": previous_user_resolved_utterance
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
                    conversation_history=conversational_history,
                    relevance_judgements=relevance_judgements,
                    user_utterance_type=utterance_type,
                    user_preferences=user_preferences
                )
