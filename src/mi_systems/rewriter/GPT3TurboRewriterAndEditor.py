import os

from src.data_classes.conversational_turn import ConversationalTurn
from dotenv import load_dotenv
from openai import OpenAI

from .AbstractRewriter import AbstractRewriter
from tenacity import retry, wait_exponential

load_dotenv()

# Rewrite step.
REWRITE_INSTRUCTION = "Given a question and its context, decontextualize the \
    question by addressing coreference and omission issues. The resulting \
    question should retain its original meaning and be as informative as \
    possible, and should not duplicate any previously asked questions in \
    the context."

REWRITE_EXAMPLE_1 = "Context: [Q: When was Born to Fly released? A: Sara \
    Evans's third studio album, Born to Fly, was released on October 10, \
    2000.]\nQuestion: Was Born to Fly well received by critics?\nRewrite: \
    Was Born to Fly well received by critics?"

REWRITE_EXAMPLE_2 = "Context: [Q: When was Keith Carradine born? A: Keith \
    Ian Carradine was born August 8, 1949. Q: Is he married? A: Keith \
    Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they \
    have any children?\nRewrite: Do Keith Carradine and Sandra Will have \
    any children?"

REWRITE_EXAMPLE_3 = "Context: [Q: Who proposed that atoms are the basic \
    units of matter? A: John Dalton proposed that each chemical element is \
    composed of atoms of a single, unique type, and they can combine to form \
    more complex structures called chemical compounds.]\nQuestion: How did \
    the proposal come about?\nRewrite: How did John Dalton's proposal that \
    each chemical element is composed of atoms of a single unique type, and \
    they can combine to form more complex structures called chemical \
    compounds come about?"

REWRITE_EXAMPLE_4 = "Context: [Q: What is it called when two liquids \
    separate? A: Decantation is a process for the separation of mixtures \
    of immiscible liquids or of a liquid and a solid mixture such as a \
    suspension. Q: How does the separation occur? A: The layer closer to the \
    top of the container-the less dense of the two liquids, or the liquid \
    from which the precipitate or sediment has settled out-is poured \
    off.]\nQuestion: Then what happens?\nRewrite: Then what happens after \
    the layer closer to the top of the container is poured off with \
    decantation?"

REWRITE_EXAMPLES = "\n\n".join(
    [REWRITE_EXAMPLE_1, REWRITE_EXAMPLE_2, REWRITE_EXAMPLE_3,
     REWRITE_EXAMPLE_4]
)

# Edit Step
EDIT_INSTRUCTION = "Given a question and its context and a rewrite that \
    decontextualizes the question, edit the rewrite to create a revised \
    version that fully addresses coreferences and omissions in the \
    question without changing the original meaning of the question but \
    providing more information. The new rewrite should not duplicate any \
    previously asked questions in the context. If there is no need to edit \
    the rewrite, return the rewrite as-is."

EDIT_EXAMPLE_1 = "Context: [Q: When was Born to Fly released? A: Sara \
    Evans's third studio album, Born to Fly, was released on October 10, \
    2000.]\nQuestion: Was Born to Fly well received by critics?\nRewrite: \
    Was Born to Fly well received by critics?\nEdit: Was Born to Fly well \
    received by critics?"

EDIT_EXAMPLE_2 = "Context: [Q: When was Keith Carradine born? A: Keith Ian \
    Carradine was born August 8, 1949. Q: Is he married? A: Keith Carradine \
    married Sandra Will on February 6, 1982.]\nQuestion: Do they have any \
    children?\nRewrite: Does Keith Carradine have any children?\nEdit: \
    Do Keith Carradine and Sandra Will have any children?"

EDIT_EXAMPLE_3 = "Context: [Q: Who proposed that atoms are the basic units \
    of matter? A: John Dalton proposed that each chemical element is \
    composed of atoms of a single, unique type, and they can combine to \
    form more complex structures called chemical compounds.]\nQuestion: How \
    did the proposal come about?\nRewrite: How did John Dalton's proposal \
    come about?\nEdit: How did John Dalton's proposal that each chemical \
    element is composed of atoms of a single unique type, and they can \
    combine to form more complex structures called chemical compounds \
    come about?"

EDIT_EXAMPLE_4 = "Context: [Q: What is it called when two liquids separate? \
    A: Decantation is a process for the separation of mixtures of immiscible \
    liquids or of a liquid and a solid mixture such as a suspension. Q: How \
    does the separation occur? A: The layer closer to the top of the \
    container-the less dense of the two liquids, or the liquid from which \
    the precipitate or sediment has settled out-is poured off.]\nQuestion: \
    Then what happens?\nRewrite: Then what happens after the layer closer \
    to the top of the container is poured off?\nEdit: Then what happens \
    after the layer closer to the top of the container is poured off with \
    decantation?"

EDIT_EXAMPLES = "\n\n".join(
    [EDIT_EXAMPLE_1, EDIT_EXAMPLE_2, EDIT_EXAMPLE_3, EDIT_EXAMPLE_4])


class GPT3TurboRewriterAndEditor(AbstractRewriter):

    def __init__(self, model_name="gpt-3.5-turbo") -> None:
        self.model_name = model_name
        self.client = OpenAI()

    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        query = conversational_turn.user_utterance
        history = self._parse_history(conversational_turn)
        rewrite = self._rewrite(query, history)
        edited_rewrite = self._edit(query, rewrite, history)
        return edited_rewrite

    def _rewrite(self, query: str, history: str) -> str:
        prompt = (
            f"{REWRITE_INSTRUCTION}\n\n{REWRITE_EXAMPLES}\n\nContext: "
            f"{history}\nQuestion: {query}\nRewrite: "
        )
        return self._get_response(prompt)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_response(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=2560,
            n=1,
        )

        return response.choices[0].message.content

    def _edit(self, query: str, rewrite: str, history: str) -> str:
        prompt = (
            f"{EDIT_INSTRUCTION}\n\n{EDIT_EXAMPLES}\n\nContext: {history}\n"
            f"Question: {query}\nRewrite: {rewrite}\nEdit: "
        )
        return self._get_response(prompt)

    def _parse_history(self, conversational_turn: ConversationalTurn) -> str:
        history = []
        for turn in conversational_turn.conversation_history:
            if turn["participant"] == "User":
                history.append(f"Q: {turn['utterance']}")
            else:
                history.append(f"A: {turn['utterance']}")
        parsed_history = " ".join(history)
        parsed_history = f"[{parsed_history}]"

        return parsed_history
