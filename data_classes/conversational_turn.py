from dataclasses import dataclass
from typing import Dict, List, NamedTuple

from ir_measures import Qrel


@dataclass
class Document:
    doc_id: str
    doc_text: str
    score: float = None


@dataclass
class ConversationalTurn(NamedTuple):
    turn_id: str  # ideally should be an int, but CAsT year 4 is a string
    information_need: str
    utterance: str
    relevance_judgements: List[Qrel]
    rewritten_utterance: str = None
    conversation_history: List[Dict[str, str]] = []
    ranking: List[Document] = None
    clarifying_question: str = None
    cq_answer: str = None
