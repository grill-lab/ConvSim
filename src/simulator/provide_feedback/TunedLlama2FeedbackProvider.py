from string import Template

from src.data_classes.conversational_turn import ConversationalTurn
from transformers import TextGenerationPipeline

from .AbstractFeedbackProvider import AbstractFeedbackProvider


class TunedLlama2FeedbackProvider(AbstractFeedbackProvider):

    def __init__(self, pipeline: TextGenerationPipeline, adapter_name: str):
        self.adapter_name = adapter_name
        self.simulator = pipeline

        with open("src/prompts/llama2_tuned_prompt.txt", "r") as f:
            self.prompt = f.read()
        
    def give_feedback(self, conversational_turn: ConversationalTurn
                        ) -> ConversationalTurn:
        
        self.simulator.model.set_adapter(self.adapter_name)
        parsed_conversation = self.__parse_conversation(conversational_turn)

        prompt = Template(self.prompt).substitute(
            information_need=conversational_turn.information_need.lower(),
            conversation=parsed_conversation)

        response = self.simulator(
            prompt, max_new_tokens=64, temperature=0.1, 
            return_full_text=False,
        )[0]["generated_text"]
        response = response.split('[/INST]')[0].strip()

        return response
    
    def __parse_conversation(self, conversational_turn: ConversationalTurn) -> str:
        """Format the conversation history into a single string for llama2"""
        conversation = ""
        for turn in conversational_turn.conversation_history:
            utterance = turn["utterance"]
            if turn["participant"] == "User":
                conversation += f"{utterance} [/INST] "
            elif turn["participant"] == "System":
                conversation += f"{utterance} </s><s>[INST] "
        
        # Add newest turns
        conversation +=  f"{conversational_turn.user_utterance} [/INST] "
        conversation += f"{conversational_turn.system_response} </s><s>[INST] "

        return conversation