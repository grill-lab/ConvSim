from typing import List
from src.data_classes.conversational_turn import ConversationalTurn
from .AbstractFeedbackProvider import AbstractFeedbackProvider
import random, json
import math


class RandomFeedbackProvider(AbstractFeedbackProvider):
    def __init__(self) -> None:
        super().__init__()
        with open('shenanigans/files/feedback_utterances.json') as f:
            self.feedback_utterances = json.load(f)
    
    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
        precision_at_3 = conversational_turn.evaluate_turn(measure='P(rel=2)@3')
        precision_at_1 = conversational_turn.evaluate_turn(measure='P(rel=2)@1')

        if precision_at_3 > 0.3 or precision_at_1 == 1.0:
            potential_feedback = self.feedback_utterances[conversational_turn.turn_id]['POSITIVE']
            if potential_feedback:
                return random.choice(potential_feedback)
            else:
                return 'Thanks!'
        
        else:
            potential_feedback = self.feedback_utterances[conversational_turn.turn_id]['NEGATIVE']
            if potential_feedback:
                return random.choice(potential_feedback)
            else:
                return 'That does not answer my question'
