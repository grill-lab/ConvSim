from dotenv import load_dotenv
from openai import OpenAI
from src.data_classes.conversational_turn import ConversationalTurn
from tenacity import retry, wait_exponential

from .AbstractFeedbackProvider import AbstractFeedbackProvider
from string import Template

load_dotenv()


class OpenAIFeedbackProvider(AbstractFeedbackProvider):
    def __init__(self):
        self.client = OpenAI()
    
    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
        feedback = self.__get_message(conversational_turn, use_ptkbs=True)

        return feedback
    
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
                      use_ptkbs: bool = False) -> str:
        conversation_str = self.__parse_conversation(conversational_turn)
        if use_ptkbs:
            system_prompt = ("The following is a conversation between a user and "
                             "a search system where the user wants to learn more "
                             "about $information_need. Here is some more information "
                             "about the user in their own words: $ptkbs. "
                             "Your job is to take the role of the user and "
                             "continue the conversation to provide direct "
                             "comments about the system's responses or provide "
                             "answers to the system's clarifying questions. ")
            prefs_str = " ".join(
                f"({idx}) {pref}" for idx, pref in enumerate(
                    conversational_turn.user_preferences, start=1
                )
            )
            system_prompt = Template(system_prompt).substitute(
                information_need=conversational_turn.information_need.lower(),
                ptkbs=prefs_str
            )
        else:
            system_prompt = ("The following is a conversation between a user and "
                            "a search system where the user wants to learn more "
                            "about $information_need. "
                            "Your job is to take the role of the user and "
                             "continue the conversation to provide direct "
                             "comments about the system's responses or provide "
                             "answers to the system's clarifying questions. "
                             "Do not respond as the system.")
            system_prompt = Template(system_prompt).substitute(
                information_need=conversational_turn.information_need.lower()
            )
        
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
            frequency_penalty=0.2,
            presence_penalty=0.5
        )

        return response.choices[0].message.content.replace("USER:", "").replace("SYSTEM:", "").replace("ASSISTANT:", "").strip()