from .AbstractModule import AbstractModule
from src.simulator.provide_feedback import AbstractFeedbackProvider
from src.mi_systems.retriever import AbstractRetriever
from src.mi_systems.rewriter import AbstractRewriter
from src.mi_systems.reranker import AbstractReranker
from src.mi_systems.response_generator import AbstractResponseGenerator
from src.mi_systems.askCQ import AbstractAskCQ

from typing import List
from src.data_classes.conversational_turn import ConversationalTurn
from transformers import TextGenerationPipeline, TextClassificationPipeline
import string
import re


class Pipeline(AbstractModule):
    """Single pass through all conversational modules"""

    def __init__(self, modules: List[AbstractModule]) -> None:
        self.modules = modules

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:
            conversational_turn = module.step(conversational_turn)

        return conversational_turn


class RecursivePipeline(Pipeline):
    """Allows for multiple feedback rounds"""

    def __init__(self, modules: List[AbstractModule], max_feedback_rounds=3, min_ndcg=0.75) -> None:
        super().__init__(modules)
        self.max_feedback_rounds = max_feedback_rounds
        self.min_ndcg = min_ndcg

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:

            if (isinstance(module, AbstractFeedbackProvider) and 
                conversational_turn.feedback_rounds >= self.max_feedback_rounds):
                continue

            conversational_turn = module.step(conversational_turn)
            if (isinstance(module, AbstractFeedbackProvider) and 
                conversational_turn.feedback_rounds < self.max_feedback_rounds):
                conversational_turn.feedback_rounds += 1
                conversational_turn = self.step(conversational_turn)

        return conversational_turn


class SingleActionLlamaAgentPipeline(AbstractModule):
    """Pipeline for single action agent"""

    def __init__(self, agent_model: TextGenerationPipeline, adapter_name: str, 
                 searcher: AbstractRetriever, rewriter: AbstractRewriter, 
                 reranker: AbstractReranker, user_simulator: AbstractFeedbackProvider,
                 response_generator: AbstractResponseGenerator) -> None:
        self.agent_model = agent_model
        self.adapter_name = adapter_name

        self.clarify_pipeline = Pipeline([user_simulator, rewriter, searcher, reranker, response_generator])
        self.search_pipeline = Pipeline([searcher, reranker, response_generator])
        self.rerank_pipeline = Pipeline([reranker, response_generator])
        self.fall_back_pipeline = Pipeline([rewriter, searcher, reranker, response_generator])


        with open("src/prompts/llama2_search_clarify_no_passages.txt") as f:
            self.search_clarify_no_passages_prompt = f.read()
        
        with open("src/prompts/llama2_search_clarify_rerank_respond_passages.txt") as f:
            self.search_clarify_respond_passages_prompt = f.read()

        self.action_pattern = re.compile(
            r'<action>(.*?)</action>', re.DOTALL)
        self.utterance_pattern = re.compile(
            r'<utterance>(.*?)</utterance>', re.DOTALL)

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        self.agent_model.model.set_adapter(self.adapter_name)
        prompt = self._build_prompt(conversational_turn)
        response = self.agent_model(
            prompt, max_new_tokens=128, temperature=0.1, 
            return_full_text=False
        )[0]["generated_text"]
        action, utterance = self._parse_output(response)
        fallback_q = "Can you provide more details or specify the context of your query to help me better understand and give you an accurate response?"
        
        if "clarify" in action or "clarifying" in action:
            question = utterance if utterance else fallback_q
            conversational_turn.update_history(
                question, participant="System", utterance_type="clarifying_question"
            )
            conversational_turn = self.clarify_pipeline(conversational_turn)
        elif action == "search":
            conversational_turn.rewritten_utterance = utterance if utterance else conversational_turn.user_utterance
            conversational_turn = self.search_pipeline(conversational_turn)
        elif action == "rerank":
            conversational_turn.rewritten_utterance = utterance if utterance else conversational_turn.user_utterance
            conversational_turn = self.rerank_pipeline(conversational_turn)
        elif action == "respond":
            response = utterance if utterance else conversational_turn.conversation_history[-1]["utterance"]
            conversational_turn.update_history(
                response, participant="System", utterance_type="response"
            )
        else:
            print("Turn ID: ", conversational_turn.turn_id)
            print(
                "Unable to parse response at turn, so falling back. here is the response: ", 
                response
            )
            conversational_turn.update_history(
                fallback_q, participant="System", utterance_type="clarifying_question"
            )
            conversational_turn = self.clarify_pipeline(conversational_turn)
        
        return conversational_turn

    def _parse_output(self, response: str) -> tuple[str, str]:
        input_string = f"<root>{response.strip()}</root>"
        
        action_match = self.action_pattern.search(input_string)
        utterance_match = self.utterance_pattern.search(input_string)

        action = ""
        utterance = ""
        if action_match:
            action = action_match.group(1).strip()
        if utterance_match:
            utterance = utterance_match.group(1).strip()
        
        return action.lower(), utterance

    def _build_prompt(self, conversational_turn: ConversationalTurn) -> str:
        conversation = ""
        for turn in conversational_turn.conversation_history:
            conversation += f"{turn['participant'].upper()}: {turn['utterance']}\n"
        
        # Add newest turns
        conversation +=  f"USER: {conversational_turn.user_utterance}\n"

        if conversational_turn.ranking:
            passage_texts = [passage.doc_text for passage in conversational_turn.ranking][:3]
            passage_texts = "\n".join(f"({idx}): {rp}" for idx, rp in enumerate(passage_texts, start=1))

            return string.Template(
                self.search_clarify_respond_passages_prompt
            ).substitute(
                conversation=conversation.strip(), 
                passages=passage_texts.strip()
            )
        
        else:
            return string.Template(
                self.search_clarify_no_passages_prompt
            ).substitute(
                conversation=conversation.strip()
            )
            
class MultiActionLlamaAgentPipeline(SingleActionLlamaAgentPipeline):
    """Pipeline for multi action agent"""

    def __init__(self, agent_model: TextGenerationPipeline, adapter_name: str, 
                 searcher: AbstractRetriever, rewriter: AbstractRewriter, 
                 reranker: AbstractReranker, user_simulator: AbstractFeedbackProvider,
                 response_generator: AbstractResponseGenerator) -> None:
        super().__init__(agent_model=agent_model, adapter_name=adapter_name,
                         searcher=searcher, rewriter=rewriter, 
                         reranker=reranker, user_simulator=user_simulator,
                         response_generator=response_generator)
        self.user_simulator = user_simulator
        self.searcher = searcher
        self.reranker = reranker

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        self.agent_model.model.set_adapter(self.adapter_name)
        prompt = self._build_prompt(conversational_turn)
        multiplier = 0.1 * (conversational_turn.feedback_rounds + (2 ** conversational_turn.feedback_rounds)) if conversational_turn.feedback_rounds > 0 else 0.1
        response = self.agent_model(
            prompt, 
            max_new_tokens=128,
            temperature=min(0.9, multiplier),
            return_full_text=False
        )[0]["generated_text"]
        conversational_turn.feedback_rounds += 1
        action, utterance = self._parse_output(response)
        print(f"Call {conversational_turn.feedback_rounds}. Got action: `{action}` and utterance: `{utterance}`")
        fallback_q = "Can you provide more details or specify the context of your query to help me better understand and give you an accurate response?"

        if conversational_turn.feedback_rounds > 3 or not action:
            print("Agent prompted 3 times, so simply calling clarify pipeline")
            conversational_turn.update_history(
                fallback_q, participant="System", utterance_type="clarifying_question"
            )
            conversational_turn = self.clarify_pipeline(conversational_turn)
            return conversational_turn

        elif "clarify" in action or "clarifying" in action or "clarification" in action:
            
            question = utterance if utterance else fallback_q
            conversational_turn.update_history(
                question, participant="System", utterance_type="clarifying_question"
            )
            conversational_turn = self.user_simulator(conversational_turn)
            return self.step(conversational_turn)
        
        elif action == "search" or "search" in action:
            conversational_turn.rewritten_utterance = utterance if utterance else conversational_turn.user_utterance
            conversational_turn = self.searcher(conversational_turn)
            return self.step(conversational_turn)
        
        elif action == "rerank" or "rerank" in action:
            conversational_turn.rewritten_utterance = utterance if utterance else conversational_turn.user_utterance
            conversational_turn = self.reranker(conversational_turn)
            return self.step(conversational_turn)
        
        elif action == "respond" or "respond" in action:
            response = utterance if utterance else conversational_turn.conversation_history[-1]["utterance"]
            conversational_turn.update_history(
                response, participant="System", utterance_type="response"
            )
            return conversational_turn


class SingleActionPipelineAgent(AbstractModule):
    """Pipeline for single action agent"""

    def __init__(self, agent_model: TextClassificationPipeline,
                 searcher: AbstractRetriever, rewriter: AbstractRewriter, 
                 reranker: AbstractReranker, user_simulator: AbstractFeedbackProvider,
                 clarifier: AbstractAskCQ,
                 response_generator: AbstractResponseGenerator) -> None:
        self.agent_model = agent_model

        self.clarify_pipeline = Pipeline([clarifier, user_simulator, rewriter, searcher, reranker, response_generator])
        self.search_pipeline = Pipeline([searcher, reranker, response_generator])
    
    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        conversation_history = self.__format_conversation(conversational_turn)
        action = self.agent_model(conversation_history)[0]["label"]

        if action == "clarify":
            print(f"Clarfy action selected for turn {conversational_turn.turn_id}")
            conversational_turn = self.clarify_pipeline(conversational_turn)
        else:
            print(f"Search action selected for turn {conversational_turn.turn_id}")
            conversational_turn = self.search_pipeline(conversational_turn)
        
        return conversational_turn
    

    def __format_conversation(self, conversational_turn: ConversationalTurn) -> str:
        """Format the conversations for the agent training."""
        # Format the conversations.
        conversation = "[CLS] "
        for turn in conversational_turn.conversation_history:
            if 'User' in turn and turn['User']:
                conversation += turn['User']
            if 'System' in turn and turn['System']:
                conversation += " [SEP] " + turn['System'] + " [SEP] "
        
        conversation += conversational_turn.user_utterance

        return conversation
