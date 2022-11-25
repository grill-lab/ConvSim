from .AbstractRewriter import AbstractRewriter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.data_classes.conversational_turn import ConversationalTurn
import torch


class T5Rewriter(AbstractRewriter):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "castorini/t5-base-canard"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> str:
        previous_utterances = [turn['utterance'] for turn in conversational_turn.conversation_history]
        previous_utterances += [conversational_turn.user_utterance]
        context = " ||| ".join(previous_utterances)

        with torch.no_grad():
            tokenized_input = self.tokenizer.encode(
                context, return_tensors="pt"
            ).to(self.device)
            output_ids = self.model.generate(
                tokenized_input,
                max_length=200,
                num_beams=4,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            ).to(self.device)

        rewrite = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return rewrite
