from .CAsTY4DataGenerator import CAsTY4DataGenerator
from src.data_classes import ConversationalTurn


class CAsTY3DataGenerator(CAsTY4DataGenerator):

    def get_turn(self) -> ConversationalTurn:
        for topic in self.topics:
            for index, turn in enumerate(topic['turn']):
                turn_id = f"{topic['number']}_{turn['number']}"
                information_need = turn.get("information_need")
                utterance = turn.get("raw_utterance")
                manual_utterance = turn.get("manual_rewritten_utterance")
                utterance_type = turn.get("utterance_type")
                relevance_judgements = [
                    qrel for qrel in self.qrels if qrel.query_id == turn_id]
                conversational_history = []
                for previous_turn in topic['turn'][:index]:
                    # extract utterance attributes
                    previous_user_utterance = previous_turn.get("raw_utterance")
                    previous_user_utterance_rewrite = previous_turn.get(
                        "automatic_rewritten_utterance")
                    previous_user_utterance_type = previous_turn.get(
                        "utterance_type")

                    # extract response attributes
                    previous_system_response = previous_turn.get("passage")
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
                
                yield ConversationalTurn(
                    turn_id=turn_id, information_need=information_need,
                    user_utterance=utterance,
                    manual_utterance=manual_utterance,
                    conversation_history=conversational_history,
                    user_utterance_type=utterance_type,
                    relevance_judgements=relevance_judgements
                )
