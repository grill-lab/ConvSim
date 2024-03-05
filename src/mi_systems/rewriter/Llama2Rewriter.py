from src.data_classes.conversational_turn import ConversationalTurn
from transformers import TextGenerationPipeline

from .AbstractRewriter import AbstractRewriter
import re


class Llama2Rewriter(AbstractRewriter):

    def __init__(self, pipeline: TextGenerationPipeline) -> None:
        
        self.rewriter = pipeline
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:

        prompt = self.__parse_conversation(conversational_turn)

        response = self.rewriter(
            prompt, max_new_tokens=64, temperature=0.1, do_sample=True,
            top_p=0.9, top_k=50, repetition_penalty=1.2, 
            return_full_text=False, prompt_lookup_num_tokens=10
        )[0]["generated_text"]

        if self.__parse_output(response):
            response = self.__parse_output(response)
        
        print(response)

        return response
    
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


        conversation = [
            {
                "role": "user",
                "content": ("The following is a conversation between a user "
                            "and a search system. Your job is to rewrite the "
                            "user's last utterance to the search system so "
                            "that it is free of ambiguity and the search "
                            "system is able to answer it better. Just "
                            "return the new query text with no explanations "
                            f"or anything else.\n\n {transcript}"
                            )
            }
        ]

        return self.rewriter.tokenizer.apply_chat_template(conversation, tokenize=False)
    
    def __parse_output(self, response: str) -> str:
        """Parse the output from llama2 to remove the prompt and any other
        unnecessary text.
        """
        response = response.replace("New Query Text", "")
        if '"' in response and re.findall(r'\"(.*?)\"', response):
            response = re.findall(r'\"(.*?)\"', response)[0]

        parts = response.split("\n")
        if len(parts) > 1:
            res = re.sub(r'\b[A-Z]+\b', '', parts[1]).strip()
            res = res.replace(":", "").strip()
        else:
            res = re.sub(r'\b[A-Z]+\b', '', parts[0]).strip()
            res = res.replace(":", "").strip()
        
        return res
