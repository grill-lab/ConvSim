import json
import os
import warnings
from typing import Dict, List

import requests

from src.data_classes.conversational_turn import ConversationalTurn
from src.mi_systems.askCQ.AbstractAskCQ import GenerateCQ


class SWATEntityExtractor:
    def __init__(self) -> None:
        self.url = "https://swat.d4science.org/salience"
        self.my_gcube_token = os.getenv("MY_GCUBE_TOKEN")

    def extract_entities_from_text(
        self, document_text: str
    ) -> List[Dict]:
        """Extracts entities and their salience from a given document string.

        Args:
            document_text: document text to extract entities from.
        Returns:
            A list of dictionaries representing entities with salience scores.
        """
        document = {"title": "", "content": document_text}

        response = requests.post(
            self.url, data=json.dumps(document), params={"gcube-token": self.my_gcube_token}
        )

        ret = []
        for item in response.json()["annotations"]:

            ret.append(
                {
                    "salience_score": item["salience_score"],
                    "entity": item["wiki_title"].replace("_", " "),
                }
            )
        return ret

    def extract_entities(self,
                         conversational_turn: ConversationalTurn,
                         top_n_docs: int = 1,
                         salience_threshold: float = 0.35,
                         n_entities: int = 3
    ) -> List[str]:
        """"Extract most salient entities from ranked list of documents.
        
        Args:
            conversational_turn: ConversationalTurn.
            top_n_docs: how many top ranked documents to extract entities from.
            salience_threshold: minimum saliency score to keep an entity.
            n_entities: number of most salient entities to return.
        Returns:
            A list of most entities as strings.
        """
        top_docs = conversational_turn.ranking[:top_n_docs]
        doc_text_concatenated = " ".join([doc.doc_text for doc in top_docs])
        entities = self.extract_entities_from_text(doc_text_concatenated)

        # select only entities above the threshold
        entities = [e for e in entities
                    if e["salience_score"] > salience_threshold]
        # sort by salience and return top n_entities
        entities = sorted(entities, 
                          key=lambda t: t["salience_score"], reverse=True)
        return entities[:n_entities]

class TemplateAskCQ(GenerateCQ):

    def __init__(self):
        super().__init__()
        self.entity_extractor = SWATEntityExtractor()

    def ask_cq(self, conversational_turn: ConversationalTurn) -> str:
        """Templated clarifying question based on entities."""

        # extract entities
        # TODO: figure out how to pass the arguments (n_entities etc.)
        entities = self.entity_extractor.extract_entities(conversational_turn)
        entities = [e["entity"] for e in entities]

        if len(entities) == 1:
            return f"Are you interested in {entities[0]}?"
        elif len(entities) == 2:
            return f"Are you interested in {entities[0]} or {entities[1]}?"
        elif len(entities) == 3: # TODO: enable more than 3 entities?
            return f"Are you interested in {entities[0]}, {entities[1]}" + \
                    f" or {entities[2]}?"
        else:
            warnings.warn(f"Number of extracted entities not in [1,2,3]." + \
                f"Returning empty clarifying question. Entities: {entities}")
            return " "



        
