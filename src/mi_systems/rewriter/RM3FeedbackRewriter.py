from src.data_classes.conversational_turn import ConversationalTurn
from .AbstractRewriter import AbstractRewriter
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import tokenize


class RM3FeedbackRewriter(AbstractRewriter):

    def __vectorize(self, documents):
        """
        Vectorises documents and returns vocabulary
        """
        vectorizer = TfidfVectorizer(tokenizer=tokenize)
        vectors = vectorizer.fit_transform(documents)
        vocabulary = vectorizer.vocabulary_

        return vectors.toarray(), vocabulary

    def __get_top_terms(self, vector, vocabulary, n=10):
        # Sort the terms by their weights in the vector
        term_weights = sorted(zip(vector, vocabulary), reverse=True)

        # Look up the terms corresponding to the top indices
        top_terms = [term[1] for term in term_weights[:n]]

        return top_terms

    def rewrite(self, conversational_turn: ConversationalTurn, alpha: int = 1,
                beta: float = 0.75, gamma: float = 0.25) -> str:
        if conversational_turn.feedback_rounds > 0:
            query = conversational_turn.conversation_history[-1]['rewritten_utterance'] if \
                conversational_turn.conversation_history[-1]['rewritten_utterance'] else \
                conversational_turn.conversation_history[-1]['utterance']
            # if conversation is going on beyond first feedback round, we
            # assume that feedback is negative
            positive_feedback = ""
            negative_feedback = conversational_turn.user_utterance

            documents = [query, positive_feedback, negative_feedback]

            # Vectorize the documents and build the vocabulary
            vectors, vocabulary = self.__vectorize(documents)

            # Extract the vectors for the query, positive feedback, and
            # negative feedback
            query_vector = vectors[0]
            positive_feedback_vector = vectors[1]
            negative_feedback_vector = vectors[2]

            new_query_vector = query_vector * alpha + \
                positive_feedback_vector * beta - negative_feedback_vector * \
                gamma

            # Get the top terms in the new query vector
            top_terms = self.__get_top_terms(new_query_vector, vocabulary)
            filtered_top_terms = [
                term for term in top_terms if term not in query.lower()]
            filtered_top_terms = ' '.join(filtered_top_terms)

            return f'{query} {filtered_top_terms}'
