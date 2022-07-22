from typing import Dict, List, Any

class AbstractOrchestrator:
    def __init__(self,
            information_need: str,
            qrels: List[str],
            initial_query: str = None,
            ):
        """ Initialise the stateful orchestrator.

            Args:
                information_need: Information need description,
                initial_query: Initial query (optional),
                qrels: A list of documents relevant to the given info need.
        """

        self.information_need = information_need
        self.qrels = qrels
        self.initial_query = initial_query

        self.conversation_history = []

    def run_turn(self, current_turn: Dict[str, Any]) -> str:
        """ Generate a next turn response given an input.

            Args:
                current_turn: dict with keys: {'input_string',
                        'provenance_docs' (optional),
                        'type_of_input'}, 
                        where 'type_of_input' is one of:
                        {'ranked_list', 'question'} # rethink

            Returns:
                Dictionary containing the response. (make it a class?)
        """

        # add current_turn to the history
        self.conversation_history.append(current_turn)

        # these are kind of like policies
        if current_turn['type_of_input'] == 'ranked_list':
            return self.check_ranked_list(current_turn['provenance_docs'])
        elif current_turn['type_of_input'] == 'question':
            return self.answer_clarifying_question(current_turn['input_string'])
        
    def start_search(self) -> str:
        """ Start search either by initial_query or predictic a query.

            Returns:
                Initial query (passed or predicted).
        """
        # TODO: implement
        return self.initial_query


    def check_ranked_list(self, ranked_list: List[str]):
        """ Compute a metric based on qrels and ranked_list and decide."""
        # TODO: implement
        if ranked_list[0] in self.qrels:
            return self.provide_good_feedback()
        else:
            return self.provide_bad_feedback()

    def provide_good_feedback(self):
        """ Provides good feedback when happy with results."""
        # TODO: implement
        return "Great, thanks!"

    def provide_bad_feedback(self):
        """ Provides bad feedback when given results are bad."""
        # TODO: implement
        return "That's not what I'm looking for."

    def answer_clarifying_question(self, clarifying_question: str):
        """ Answers given clarifying question in line with info need."""
        # TODO: implement
        return "No, " + self.information_need
