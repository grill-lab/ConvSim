from src.data_classes import ConversationalTurn
from .AbstractConversationalDataGenerator import (
    AbstractConversationalDataGenerator
)
import json
from src.data_classes import ConversationalTurn


class CanardDataGenerator(AbstractConversationalDataGenerator):

    def __init__(self, dataset_path: str):
        with open(dataset_path) as canard_topics_file:
            self.samples = json.load(canard_topics_file)
        
    def get_turn(self) -> ConversationalTurn:
        for idx, sample in enumerate(self.samples, start=1):
            if idx == 1000:
                break
            information_need = sample["Rewrite"]
            utterance = sample["Question"]
            turn_id = sample["QuAC_dialog_id"] + "-" + str(sample["Question_no"])
            rewrite = sample["Rewrite"]
            conversation_history = []
            user_utterance_type = "question"
            for i, turn in enumerate(sample["History"]):
                if i % 2 == 0:
                    conversation_history.append({
                        "participant": "User",
                        "utterance": turn,
                        "utterance_type": "question"
                    })
                else:
                    conversation_history.append({
                        "participant": "System",
                        "utterance": turn,
                        "utterance_type": "response"
                    })
            # Fake judgements.
            relevance_judgements = [f"a{i}-{i-1}" for i in range(500)]
            yield ConversationalTurn(
                turn_id=turn_id, information_need=information_need,
                user_utterance=utterance, rewritten_utterance=rewrite,
                conversation_history=conversation_history,
                relevance_judgements=relevance_judgements,
                user_utterance_type=user_utterance_type
            )