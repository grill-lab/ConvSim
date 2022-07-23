from abc import ABC, abstractmethod

class AbstractSimulatedUser(ABC):

    def __init__(self, topic: str, facets: list, prior_knowledge: list, patience: int) -> None:
        self.topic = topic
        self.facets = facets
        self.prior_knowledge = prior_knowledge
        self.patience = patience
    
    def generate_utterance(self, conversation_history: list) -> str:
        pass