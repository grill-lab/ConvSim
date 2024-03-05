from dotenv import load_dotenv
from openai import OpenAI
from src.data_classes.conversational_turn import ConversationalTurn
from tenacity import retry, wait_exponential

from .AbstractRewriter import AbstractRewriter
from string import Template

load_dotenv()


class OpenAIFeedbackRewriter(AbstractRewriter):
    def __init__(self):
        self.client = OpenAI()
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        rewrite = self.__get_message(conversational_turn, 
                                     use_information_needs=True)

        return rewrite
    
    def __parse_conversation(self, conversational_turn: ConversationalTurn) -> str:
        """Format the conversations for inference."""
        conversation = ""
        for turn in conversational_turn.conversation_history:
            if turn['participant'] == 'User':
                conversation += "USER: " + turn['utterance'] + "\n"
            if turn['participant'] == 'System':
                conversation += "SYSTEM: " + turn['utterance'] + "\n"
        conversation += "USER: " + conversational_turn.user_utterance + "\n"
        return conversation

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def __get_message(self, conversational_turn: ConversationalTurn, 
                      use_information_needs:bool = False,
                      use_ptkbs:bool=False) -> str:
        conversation_str = self.__parse_conversation(conversational_turn)
        if use_information_needs and use_ptkbs:
            system_prompt = ("The following is a conversation between a "
                             "user and a search system where the user "
                             "wants to learn more about $information_need. "
                             "Here is some more information about the "
                             "user in their own words: $ptkbs "
                             "Your job is to rewrite the user's last "
                             "utterance to the system so that it is in "
                             "line with the user's information need and "
                             "preferences, free of ambiguity and clear "
                             "enough so that the search system is able to "
                             "answer it better. Just return the new query "
                             "text and nothing else.")
            prefs_str = " ".join(
                f"({idx}) {pref}" for idx, pref in enumerate(
                    conversational_turn.user_preferences, start=1
                )
            )
            system_prompt = Template(system_prompt).substitute(
                information_need=conversational_turn.information_need.lower(),
                ptkbs=prefs_str
            )
        elif use_information_needs:
            system_prompt = ("The following is a conversation between a "
                             "user and a search system where the user wants to "
                             "learn about $information_need. Your job is to rewrite "
                             "the user's last utterance to the system so that it "
                             "is free of ambiguity and the search system is "
                             "able to answer it better. Just return the new "
                             "query text and nothing else.")
            system_prompt = Template(system_prompt).substitute(
                information_need=conversational_turn.information_need.lower()
            )
        else:
            system_prompt = ("The following is a conversation between a "
                            "user and a search system. Your job is to rewrite "
                            "the user's last utterance to the system so that it "
                            "is free of ambiguity and the search system is "
                            "able to answer it better. Just return the new "
                            "query text and nothing else.")

        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": conversation_str
                }
            ],
            temperature=0.1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response.choices[0].message.content.replace("USER: ", "").replace("SYSTEM:", "").strip()