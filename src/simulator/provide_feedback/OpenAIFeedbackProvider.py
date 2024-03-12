from dotenv import load_dotenv
from openai import OpenAI
from src.data_classes.conversational_turn import ConversationalTurn
from tenacity import retry, wait_exponential

from .AbstractFeedbackProvider import AbstractFeedbackProvider
from string import Template

load_dotenv()

INSTRUCTION = "Take the role of the User and generate a response to the \
    System based on the conversation and information needs:\n\n"

INSTRUCTION_PTKBS = "Take the role of the User and generate a response to the \
    System based on the conversation and information needs. Here is some more \
    information about the user in their own words: $ptkbs\n\n"

EXAMPLES = (
    "Generate a response to the System based on the conversation and "
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
    "Information need: How to fix a Genie StealthDrive Connect 7155 "
    "Garage Door Opener.\n\n"
    "User: How do you know when your garage door opener is going bad?\n"
    "System: What type of garage door do you have?\n"
    "User: I have a Genie StealthDrive Connect 7155 Garage "
    "Door Opener.\n\n"
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
    "Information need: Search engine racial bias\n\n"
    "User: In what ways can search engines be biased?\n"
    "System: Search engines can be biased in a number of different "
    "ways such as financial, political, ideological, racial, "
    "sexual, ethical, etc. Are you interested in a specific way?\n"
    "User: Yes, I am interested in the racial way.\n\n"
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
    "Information need: Micheal Jordan's NBA records across "
    "history\n\n"
    "User: I'm interested in NBA records.\n"
    "System: Where are you from?\n"
    "User: That's not relevant my question. I want to know Micheal "
    "Jordan's NBA records across history.\n\n"
    "----------\n"
)


class OpenAIFeedbackProvider(AbstractFeedbackProvider):
    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.client = OpenAI()
        self.model = model_name

    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
        feedback = self.__get_message(conversational_turn, use_ptkbs=True)

        return feedback

    def __parse_conversation(self,
                             conversational_turn: ConversationalTurn) -> str:
        """Format the conversations for inference."""
        parsed_history = ""
        for turn in conversational_turn.history:
            if turn["participant"] == "User":
                parsed_history += f'User: {turn["utterance"]}\n'
            elif turn["participant"] == "System":
                parsed_history += f'System: {turn["utterance"]}\n'

        parsed_history += f"User: {conversational_turn.user_utterance}\n"
        parsed_history += f"System: {conversational_turn.system_response}\n"

        return parsed_history

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def __get_message(self, conversational_turn: ConversationalTurn,
                      use_ptkbs: bool = False) -> str:
        conversation_str = self.__parse_conversation(conversational_turn)
        if use_ptkbs:
            prefs_str = " ".join(
                f"({idx}) {pref}" for idx, pref in enumerate(
                    conversational_turn.user_preferences, start=1
                )
            )

            instruction = Template(INSTRUCTION_PTKBS).substitute(
                ptkbs=prefs_str
            )
        else:
            instruction = INSTRUCTION
        
        information_need = conversational_turn.information_need.lower()
        prompt = (
            f"{instruction}"
            f"{EXAMPLES}"
            f"Information need: {information_need}\n\n"
            f"{conversation_str}"
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=256,
            n=1,
        )

        return response.choices[0].message.content
