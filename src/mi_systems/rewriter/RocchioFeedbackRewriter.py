from src.data_classes.conversational_turn import ConversationalTurn
from .AbstractRewriter import AbstractRewriter
from .utils import tokenize
import collections


class LanguageModel:
    def __init__(self, words):
        self.words = words
        self.build_language_model()

    def build_language_model(self):
        self.lm_counter = collections.Counter()
        for word in self.words:
            self.lm_counter[word] += 1
        self.total_words = sum(self.lm_counter.values())

    def score(self, word):
        probability = self.lm_counter[word] / (self.total_words + 1)
        return probability


class RocchioFeedbackRewriter(AbstractRewriter):

    def rewrite(self, conversational_turn: ConversationalTurn,
                alpha: float = 0.5, num_terms: int = 10) -> str:
        if conversational_turn.feedback_rounds > 0:
            query = conversational_turn.conversation_history[-1]['utterance']
            feedback = conversational_turn.user_utterance

            potential_expansion_terms = tokenize(feedback)
            query_terms = tokenize(query)

            feedback_lm = LanguageModel(potential_expansion_terms)
            query_lm = LanguageModel(query_terms)

            term_weights = {}
            for feedback_term in potential_expansion_terms:
                prob_feedback_term = feedback_lm.score(feedback_term)
                query_probability = 1
                for query_term in query_terms:
                    query_probability *= feedback_lm.score(query_term)
                prob_term_given_feedback = prob_feedback_term * query_probability
                term_weights[feedback_term] = prob_term_given_feedback

            for term, score in term_weights.items():
                final_term_score = (alpha * score) + \
                    ((1-alpha) * query_lm.score(term))
                term_weights[term] = final_term_score

            # sort weights and add num_terms to query
            term_weights = {term: score for term, score in sorted(
                term_weights.items(), key=lambda ele: ele[1], reverse=True
            )}
            expansion_terms = list(term_weights.keys())[:num_terms]

            rewrite = query + f" {' '.join(expansion_terms)}"
            return rewrite
