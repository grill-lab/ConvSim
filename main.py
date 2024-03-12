"""Entry Point for Experiments"""
import json
import shelve
from pathlib import Path

from experimental_pipelines import standard_baseline_with_openai_simulator as pipeline
from src.data_generators import IKATDataGenerator
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()

run_name = "T5Rewriter+BM25+T5Ranker+T5ResponseGenerator+OpenAIFeedbackProvider"
base_path = "data/generated_conversations"
output_path = f"{base_path}/{run_name}/transcripts"

data_generator = IKATDataGenerator(
    dataset_path="../../data/ikat/data/queries/2023_test_topics.json",
    relevance_judgements_path=(
        "../../data/ikat/data/queries/2023-qrels.all-turns.txt")
)

pipeline = pipeline()

Path(output_path).mkdir(parents=True, exist_ok=True)
existing_transcripts = [f.stem for f in Path(output_path).iterdir()]

with shelve.open(f"{base_path}/{run_name}/turns_db") as db:
    for conversational_turn in tqdm(data_generator.get_turn(), total=332):
        # Skip if already processed to save time.
        if conversational_turn.turn_id in db and conversational_turn.turn_id in existing_transcripts:
            continue
        conversational_turn = pipeline(conversational_turn)
        turn_transcript = [
            {
                "participant": turn["participant"],
                "utterance": turn["utterance"],
                "type": turn["utterance_type"]
            } for turn in conversational_turn.conversation_history
        ]

        user_dict = {
            "participant": "User",
            "utterance": conversational_turn.user_utterance,
            "type": conversational_turn.user_utterance_type
        }

        system_dict = {
            "participant": "System",
            "utterance": conversational_turn.system_response,
            "type": conversational_turn.system_response_type
        }

        if len(conversational_turn.conversation_history) % 2 == 0:
            turn_transcript.extend([user_dict, system_dict])
        else:
            turn_transcript.extend([system_dict, user_dict])

        with open(f"{output_path}/{conversational_turn.turn_id}.json", "w",
                  encoding="utf-8") as f:
            json.dump(turn_transcript, f, indent=4, ensure_ascii=False)

        db[conversational_turn.turn_id] = conversational_turn

        with open(f"{base_path}/{run_name}/{run_name}.run", "a") as f:
            for index, document in enumerate(conversational_turn.ranking):
                f.write(
                    f"{conversational_turn.turn_id}\tQ0\t{document.doc_id}\t{index+1}\t{document.score}\t{run_name}\n")
