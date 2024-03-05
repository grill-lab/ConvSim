from .AbstractRewriter import AbstractRewriter
from src.data_classes.conversational_turn import ConversationalTurn

from pyserini.search.lucene import LuceneSearcher


class FeedbackRewriter(AbstractRewriter):

    def __init__(self, collection, feedback_type='rocchio'):
        self.retriever = LuceneSearcher(collection)
        if feedback_type == 'rocchio':
            self.retriever.set_rocchio()
        elif feedback_type == 'rm3':
            self.retriever.set_rm3()
        self.retriever.set_bm25(4.46, 0.82)
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        previous_utterances = [turn['utterance'] for turn in conversational_turn.conversation_history]
        previous_utterances = ' '.join(previous_utterances)
        # feedback
        current_utterance = conversational_turn.user_utterance

        feedback_terms = self.retriever.get_feedback_terms(
            f'{previous_utterances} {current_utterance}'
        )
        feedback_terms = {term: score for term, score in sorted(
            feedback_terms.items(), key=lambda ele: ele[1], reverse=True
        )}
        feedback_terms = list(feedback_terms.keys())[:5]

        rewrite = current_utterance + f" {' '.join(feedback_terms)}"
        return rewrite



    
