from dataclasses import dataclass
from typing import List, NamedTuple, Dict
from ir_measures import Qrel


@dataclass
class ConversationalTurn(NamedTuple):
    turn_id: str  # ideally should be an int, but CAsT year 4 is a string
    information_need: str
    utterance: str
    conversation_history: List[Dict[str, str]]
    relevance_judgements: List[Qrel]
