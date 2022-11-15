from dataclasses import dataclass, field
from typing import Dict, List

from ir_measures import Qrel


@dataclass
class Document:
    doc_id: str
    doc_text: str
    score: float = None


@dataclass
class ConversationalTurn:
    turn_id: str  # ideally should be an int, but CAsT year 4 is a string
    information_need: str
    user_utterance: str
    relevance_judgements: List[Qrel]
    rewritten_utterance: str = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    ranking: List[Document] = None
    system_response: str = None
    system_response_type: str = None  # "CQ" response (text)"

    def update_history(
        self, utterance: str, participant: str, utterance_type: str = None
    ) -> None:
        """Update history, utterance_type only valid when participant is System
        """
        # it doesn't store initial query to history
        if participant == "User":
            self.conversation_history += [{"User": self.user_utterance}]
            self.user_utterance = utterance
        elif participant == "System":
            self.conversation_history += [{"System": self.system_response}]
            self.system_response = utterance
            self.system_response_type = utterance_type

    def evaluate_turn(self):
        pass
