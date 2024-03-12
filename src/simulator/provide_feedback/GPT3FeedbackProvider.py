from typing import List
from src.data_classes.conversational_turn import ConversationalTurn
from .AbstractFeedbackProvider import AbstractFeedbackProvider
from src.simulator.utils import ping_GPT3


class GPT3FeedbackProvider(AbstractFeedbackProvider):
    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
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
            "Give feedback to the SYSTEM based on the conversation and "
            "information needs:\n\n"
            "Examples:\n"
            "----------\n"
            "Information need: The effects of climate changes.\n\n"
            "User: I remember Glasgow hosting COP26 last year, but "
            "unfortunately I was out of the loop. What was it about?\n"
            "System: The COP26 event is a global united Nations summit "
            "about climate change and how countries are planning to tackle it."
            " The term “climate change” is often used as if it means the same"
            " thing as the term “global warming”. The National Academy of "
            "Sciences says “climate change” is growing in favor of "
            "“global warming” because it helps convey that there are other "
            "changes in addition to rising temperatures. In fact, "
            "“climate change” means major changes in temperature, rainfall, "
            "snow, or wind patterns lasting for decades or longer.\n"
            "User: Interesting. What are the effects of these changes?\n"
            "System: Climate change is very likely having an impact now on "
            "our planet and its life, according to the latest instalment of "
            "a report published by the Intergovernmental Panel on Climate "
            "Change (IPCC). And the future problems caused by rising seas, "
            "growing deserts, and more frequent droughts all look set to "
            "affect the developing world more than rich countries, they add.\n"
            "User: That’s rather vague. Can you be more specific?\n\n"
            "----------\n"
            "Information need: Making beauty pamper packs at home.\n\n"
            "User: I’d like to appreciate my mom by making her a pamper pack."
            " What do you put in one?\n"
            "System: Beauty Products. Mother’s Day is a day to encourage "
            "your mom to pamper herself. There’s no better way to do this "
            "than with beauty products! You can fill her basket with those "
            "if you know what her favourites are—otherwise, select universal "
            "things, such as body cream and bubble baths.\n"
            "User: Can I make them at home?\n"
            "System: Yes. You can make body scrubs, moisturizers, and "
            "make-up removers at home. Best of all, the ingredients you "
            "need, such as brown sugar, honey and olive oil, are all easy "
            "to come by. It might interest you to know that olive oil is the "
            "simplest, purest moisturizer in the world. It can be used "
            "anywhere on your body, including the extra dry spots like your "
            "elbows, knees and feet. It is also an excellent facial "
            "moisturizer, and if you decide to use it on your face, keep in "
            "mind that very little olive oil is needed. Too much will leave "
            "your skin feeling greasy and you may smell like an Italian "
            "restaurant.\n"
            "User: Thank you!\n\n"
            "----------\n"
            "Information need: Cultural significance of the Argentine Tango\n\n"
            "User: Tell me more about the tango.\n"
            "System: What kind of clothing women should wear for dancing "
            "Tango? A nice option for Tango clothing: Dress at knee-length "
            "combined with leggings. Many tango ladies today wear long wide "
            "trousers with slits and/or trousers with ribbons swinging "
            "around the ankles. This looks nice but time and again creates "
            "problems while dancing Tango because they get easily caught "
            "in their own heels. Especially when the heels are high. "
            "Actually, I see it happen almost every time I go out dancing "
            "Tango.\n"
            "User: Thanks, but I’m not asking about clothing. I want to "
            "know more information about the argentine dance.\n\n"
            "----------\n"
            f"Information need: {information_need}\n\n"
            f"{concatenated_history}"
            "User:"
        )

        return prompt
