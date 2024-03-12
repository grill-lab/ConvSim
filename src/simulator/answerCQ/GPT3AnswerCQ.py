from typing import List
from src.data_classes.conversational_turn import ConversationalTurn
from .AbstractAnswerCQ import AbstractAnswerCQ
from src.simulator.utils import ping_GPT3


class GPT3AnswerCQ(AbstractAnswerCQ):
    def answer_cq(self, conversational_turn: ConversationalTurn) -> str:
        prompt = self.create_prompt(
            conversational_turn.information_need, 
            conversational_turn.conversation_history, 
            conversational_turn.user_utterance, 
            conversational_turn.system_response
        )

        return ping_GPT3(prompt)

    @staticmethod
    def create_prompt(
        information_need: str,
        history: List[str],
        current_user_turn: str,
        current_system_response: str,
    ) -> str:
        concatenated_history = ""
        for turn in history:
            if turn["participant"] == "User":
                concatenated_history += f'User: {turn["utterance"]}\n'
            elif turn["participant"] == "System":
                concatenated_history += f'System: {turn["utterance"]}\n'

        concatenated_history += f"User: {current_user_turn}\n"
        concatenated_history += f"System: {current_system_response}\n"

        prompt = (
            "Generate a response to the system question based on the "
            "conversation and information needs:\n\n"
            "Examples:\n"
            "----------\n"
            "Information need: How to fix a Genie StealthDrive Connect 7155 "
            "Garage Door Opener.\n\n"
            "User: How do you know when your garage door opener is going bad?\n"
            "System: What type of garage door do you have?\n"
            "User: I have a Genie StealthDrive Connect 7155 Garage "
            "Door Opener.\n\n"
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
