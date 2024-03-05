from string import Template

from src.data_classes.conversational_turn import ConversationalTurn
from transformers import TextGenerationPipeline

from .AbstractFeedbackProvider import AbstractFeedbackProvider
import re


class PromptedLlama2FeedbackProvider(AbstractFeedbackProvider):

    def __init__(self, pipeline: TextGenerationPipeline) -> None:
        
        self.simulator = pipeline
        with open("src/prompts/llama2_prompted_prompt.txt", "r") as f:
            self.prompt = f.read()
        
    def give_feedback(self, conversational_turn: ConversationalTurn
                      ) -> ConversationalTurn:
        
        # parsed_conversation = self.__parse_conversation(conversational_turn)
        # parsed_preferences = self.__parse_preferences(conversational_turn)

        # prompt = Template(self.prompt).substitute(
        #     information_need=conversational_turn.information_need.lower(),
        #     preferences=parsed_preferences,
        #     conversation=parsed_conversation)

        prompt = self.__parse_conversation(conversational_turn)

        response = self.simulator(
            prompt, max_new_tokens=64, temperature=0.6, do_sample=True,
            top_p=0.9, top_k=50, repetition_penalty=1.2, 
            return_full_text=False, prompt_lookup_num_tokens=10
        )[0]["generated_text"]
        response = self.__parse_output(response)
        response = self.__parse_output(response)
        # response = response.split('[/INST]')[0].strip()

        return response
    
    # def __parse_conversation(self, conversational_turn: ConversationalTurn) -> str:
    #     """Format the conversation history into a single string for llama2"""
    #     conversation = "Hey there. I'm your personal assistant. How can I help? [/INST]"
    #     for turn in conversational_turn.conversation_history:
    #         utterance = turn["utterance"]
    #         if turn["participant"] == "User":
    #             conversation += f"{utterance} </s><s>[INST] "
    #         elif turn["participant"] == "System":
    #             conversation += f"{utterance} [/INST] "
        
    #     # Add newest turns
    #     conversation +=  f"{conversational_turn.user_utterance} </s><s>[INST] "
    #     conversation += f"{conversational_turn.system_response} [/INST] "

    #     return conversation

    def __parse_conversation(self, conversational_turn: ConversationalTurn) -> str:
        """Format the conversation history into a single string for llama2"""
        transcript = ""
        for turn in conversational_turn.conversation_history:
            utterance = turn["utterance"]
            if turn["participant"] == "User":
                transcript += f"USER: {utterance}\n"
            elif turn["participant"] == "System":
                transcript += f"SYSTEM: {utterance}\n"
        
        # Add newest turns
        transcript +=  f"USER: {conversational_turn.user_utterance}\n"
        transcript += f"SYSTEM: {conversational_turn.system_response}\n"


        conversation = [
            {
                "role": "user",
                "content": ("The following is a conversation between a user "
                            "and a search system where the user wants to "
                            f"learn about {conversational_turn.information_need.lower()}. "
                            "These are the user's preferences in their own "
                            f"words: {self.__parse_preferences(conversational_turn)}"
                            "Your job is to take the role of the user and "
                            "continue the conversation to provide direct "
                            "comments about the system's responses or provide "
                            "answers to the system's clarifying questions. "
                            "Start with 'USER:' \n\n"
                            f"{transcript}"
                            )
            }
        ]

        return self.simulator.tokenizer.apply_chat_template(conversation, tokenize=False)
    
    def __parse_preferences(self, conversational_turn: ConversationalTurn) -> str:
        """Format the user preferences into a single string for llama2"""
        return " ".join(
            f"({idx}): {up}" for idx, up in enumerate(
                conversational_turn.user_preferences,
                start=1
            )
        )
    
    def __parse_output(self, output: str) -> str:
        """Parse the output from llama2 to remove the prompt and any other
        unnecessary text.
        """
        parts = output.split("\n")
        res = parts[0].replace("USER:", "").replace("SYSTEM:", "").strip()
        # res = re.sub(r'\b[A-Z]+\b', '', parts[0]).strip()
        res = res.replace(":", "").strip()
        return res
