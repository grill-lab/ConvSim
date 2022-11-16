import os
from typing import List

import openai
from dotenv import load_dotenv

from src.base_module.AbstractModule import AbstractModule
from src.data_classes.conversational_turn import ConversationalTurn

from .AbstractAnswerCQ import AbstractAnswerCQ

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class GPT3AnswerCQ(AbstractAnswerCQ, AbstractModule):
    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        response = self.answer_cq(
            conversational_turn.information_need,
            conversational_turn.conversation_history,
            conversational_turn.user_utterance,
            conversational_turn.system_response,
        )

        conversational_turn.update_history(response, "User")
        return conversational_turn

    def answer_cq(
        self,
        information_need: str,
        history: List[str],
        current_user_turn: str,
        current_system_response: str,
    ) -> str:
        prompt = self.create_prompt(
            information_need, history, current_user_turn, current_system_response
        )
        print(prompt)

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.5,
            temperature=0.5,
        )

        return response.choices[0].text.strip()

    @staticmethod
    def create_prompt(
        information_need: str,
        history: List[str],
        current_user_turn: str,
        current_system_response: str,
    ) -> str:
        concatenated_history = ""
        for turn in history:
            if "User" in turn.keys():
                concatenated_history += f'User: {turn["User"]}\n'
            elif "System" in turn.keys():
                concatenated_history += f'System: {turn["System"]}\n'

        concatenated_history += f"User: {current_user_turn}\n"
        concatenated_history += f"System: {current_system_response}\n"

        prompt = (
            "Generate a response to the system question based on the "
            "conversation and information needs:\n\n"
            "Examples:\n"
            "----------\n"
            "Information need: History of Bees\n\n"
            "User: Tell me something interesting about bees.\n"
            "System: There are lots of interesting things to say about bees. "
            "Would you like to know about it's evolution, characteristics, "
            "or sociality?\n"
            "User: No, tell me something interesting about the history of "
            "bees.\n\n"
            "----------\n"
            "Information need: How to fix a Genie StealthDrive Connect 7155 "
            "Garage Door Opener.\n\n"
            "User: How do you know when your garage door opener is going bad?\n"
            "System: What type of garage door do you have?\n"
            "User: I have a Genie StealthDrive Connect 7155 Garage "
            "Door Opener.\n\n"
            "----------\n"
            "Information need: Understanding the effects of breast cancer.\n\n"
            "User: How deadly is it?\n"
            "System: I'm sorry, I've lost the context of the conversation. "
            "How deadly is what?\n"
            "User: I want to know how deadly breast cancer is.\n\n"
            "----------\n"
            "Information need: Search engine racial bias\n\n"
            "User: In what ways can search engines be biased?\n"
            "System: Search engines can be biased in a number of different "
            "ways such as financial, political, ideological, racial, "
            "sexual, ethical, etc. Are you interested in a specific way?\n"
            "User: Yes, I am interested in the racial way.\n\n"
            "----------\n"
            "Information need: Micheal Jordan's NBA records across "
            "history\n\n"
            "User: I'm interested in NBA records.\n"
            "System: Where are you from?\n"
            "User: That's not relevant my question. I want to know Micheal "
            "Jordan's NBA records across history.\n\n"
            "----------\n"
            f"Information need: {information_need}\n\n"
            f"{concatenated_history}"
            "User:"
        )

        return prompt
