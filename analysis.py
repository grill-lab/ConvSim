import shelve
from src.data_classes.conversational_turn import Document, ConversationalTurn
from typing import List

run_name = 'T5+BM25+T5+BART+T5-Zero-Shot'

with shelve.open('data/generated_files/utterance-picker/runs') as runs:
    conversational_turn_objects: List[ConversationalTurn] = runs[run_name]
    for ct_obj in conversational_turn_objects:
        with open(f'shenanigans/analysis/{run_name}/{ct_obj.turn_id}.txt', 'w') as f:
            for turn in ct_obj.conversation_history:
                if 'participant' in turn and 'rewritten_utterance' in turn:
                    f.write(f"{turn['utterance']} {turn['rewritten_utterance']}\n")
            f.write(f"{ct_obj.user_utterance} {ct_obj.rewritten_utterance}\n")