from .AbstractRewriter import AbstractRewriter
from src.data_classes.conversational_turn import ConversationalTurn
from transformers import pipeline
import torch


class QuReTeCRewriter(AbstractRewriter):

    def __init__(self):
        self.rewriter = pipeline(
            "ner", 
            model="uva-irlab/quretec", 
            device=-1 if not torch.cuda.is_available() else 0
        )
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        if conversational_turn.feedback_rounds > 0:
            query = conversational_turn.conversation_history[-1]['utterance']
            previous_utterances = [turn['utterance'] for turn in conversational_turn.conversation_history[:-1]]
            previous_utterances_with_feedback = " ".join(previous_utterances + [conversational_turn.user_utterance])
            context = f'[CLS] {previous_utterances_with_feedback} [SEP] {query}'

            output = self.rewriter(context)
            parsed_words = [out['word'] for out in output if out['entity'] == 'REL']
            # parsed_words = [word for word in parsed_words if word not in query.lower()] #?
            filtered_words = ' '.join(parsed_words)

            return f'{query} {filtered_words}'
        
        else:
            previous_utterances = [turn['utterance'] for turn in conversational_turn.conversation_history]
            context = f'[CLS] {previous_utterances} [SEP] {conversational_turn.user_utterance}'

            output = self.rewriter(context)
            parsed_words = [out['word'] for out in output if out['entity'] == 'REL']
            # parsed_words = [word for word in parsed_words if word not in conversational_turn.user_utterance.lower()] #?
            filtered_words = ' '.join(parsed_words)

            return f'{conversational_turn.user_utterance} {filtered_words}'