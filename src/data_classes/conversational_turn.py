from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple

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

    def update_history(self, system_response, system_response_type):
        """Update history based on current utterances (after system)."""
        # it doesn't store initial query to history
        if self.system_response is not None:
            self.conversation_history += [
                {"User": self.user_utterance},
                {"System": self.system_response},
            ]
        self.system_response = system_response
        self.system_response_type = system_response_type
    
    def evaluate_turn(self):
        pass
