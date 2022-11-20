import os
from typing import List

import openai
from dotenv import load_dotenv

from src.data_classes.conversational_turn import ConversationalTurn

from .AbstractFeedbackProvider import AbstractFeedbackProvider

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class GPT3FeedbackProvider(AbstractFeedbackProvider):
    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
        prompt = self.create_prompt(
            conversational_turn.information_need,
            conversational_turn.conversation_history,
            conversational_turn.user_utterance,
            conversational_turn.system_response
        )

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
            "Information need: Whether developed countries are meeting their"
            " obligations of providing climate finance.\n\n"
            "User: I remember Glasgow hosting COP26 last year, but"
            " unfortunately I was out of the loop. What was it about?\n"
            "System: The COP26 event is a global united Nations summit about"
            " climate change and how countries are planning to tackle it. "
            "The term “climate change” is often used as if it means the same "
            "thing as the term “global warming”. The National Academy of "
            "Sciences says “climate change” is growing in favor of “global "
            "warming” because it helps convey that there are other changes "
            "in addition to rising temperatures. In fact, “climate change” "
            "means major changes in temperature, rainfall, snow, or wind "
            "patterns lasting for decades or longer.\n"
            "User: Interesting. What are the effects of these changes?\n"
            "System: Climate change is very likely having an impact now on"
            " our planet and its life, according to the latest instalment of"
            " a report published by the Intergovernmental Panel on Climate "
            "Change (IPCC). And the future problems caused by rising seas, "
            "growing deserts, and more frequent droughts all look set to "
            "affect the developing world more than rich countries, they add.\n"
            "User: That’s interesting. Tell me more.\n"
            "System: For several years, there have been concerns that climate"
            " change negotiations will essentially ignore a key principle of"
            " climate change negotiation frameworks: the common but"
            " differentiated responsibilities. Realizing that greenhouse"
            " emissions remain in the atmosphere for a very long time, this"
            " principle recognizes that historically: Industrialized nations"
            " have emitted far more greenhouse gas emissions (even if some"
            " developing nations are only now increasing theirs); Rich "
            "countries, therefore, face the biggest responsibility and burden"
            " for action to address climate change; and Rich countries, "
            "therefore, must support developing nations adapt—through "
            "financing and technology transfer, for example. This notion of "
            "climate justice is typically ignored by many rich nations and "
            "their mainstream media, making it easy to blame China, India and"
            " other developing countries for failures in climate change "
            "mitigation negotiations.\n"
            "User: Okay, but how does it affect developing countries?\n"
            "System: Developing countries are the least able to adapt to "
            "climate change (and therefore called "
            "\"highly climate vulnerable\") due to their low levels of "
            "wealth, technology, education, and access to resources. Climate "
            "change causes 400,000 deaths on average each year, mainly due "
            "to hunger and diseases in developing countries. Interestingly, "
            "many people in developing nations who had never been educated "
            "about climate change have noticed local changes in temperature "
            "and precipitation patterns according to a study published in "
            "the journal Nature Climate Change.\n"
            "User: How are developed countries helping with that?\n"
            "System: Well, according to the Paris Agreement, every two "
            "years, developed countries are to communicate the projected "
            "levels of public climate finance they provide to developing "
            "countries to mitigate and adapt to climate change. Some "
            "developing countries also contribute to climate finance too. "
            "Already, Brazil’s President Dilma Rousseff said the country is "
            "considering contributing to climate finance, joining other "
            "emerging economies like China, which pledged to provide $3.1 "
            "billion over three years.\n"
            "User: Are they meeting them?\n"
            "System: Developed countries will communicate every two years on "
            "projected levels of public climate finance to be provided to "
            "developing countries. Developing countries will report on "
            "climate finance they will provide voluntarily. Developed "
            "countries are not obligated to provide climate finance, but "
            "they are welcome to contribute financing for climate activities.\n"
            "User: That’s not too relevant to my question.\n\n"
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
