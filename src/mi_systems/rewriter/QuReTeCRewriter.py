from src.data_classes.conversational_turn import ConversationalTurn
from transformers import pipeline

from .AbstractRewriter import AbstractRewriter

class QuReTeCRewriter(AbstractRewriter):

    def __init__(self):
        self.rewriter = pipeline(task="token-classification", model="uva-irlab/quretec")
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        turns = [turn['utterance'] for turn in conversational_turn.conversation_history]
        turns += [conversational_turn.user_utterance]
        prompt = " [SEP] ".join(turns)
        response = self.rewriter(prompt)

        relevant_terms = self.__get_relevant_words(response, conversational_turn.user_utterance)
        return conversational_turn.user_utterance + " " + relevant_terms
    
    def __get_relevant_words(self, entities, sentence):
        if not entities:
            return ""
        
        entities = sorted(entities, key=lambda x: x['start'])

        merged = [entities[0]]

        for current in entities[1:]:
            last = merged[-1]

            if current['start'] == last['end']:
                last['end'] = current['end']
            else:
                merged.append(current)
        
        words = []
        for item in merged:
            words.append(sentence[item['start']:item['end']])
        
        return " ".join(words)
