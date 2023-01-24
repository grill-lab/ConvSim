from src.base_module.Pipelines import Pipeline, RecursivePipeline
from src.data_generators import CAsTY4DataGenerator
from src.data_classes.conversational_turn import Document
from src.simulator.answerCQ import GPT3AnswerCQ
from src.simulator.provide_feedback import GPT3FeedbackProvider, RandomFeedbackProvider
from src.mi_systems.retriever import SparseRetriever
from src.mi_systems.reranker import T5Ranker, T5FeedbackRanker, CrossEncoderFeedbackRanker, CrossEncoderFeedbackRankerv2, FirstPassT5Ranker, T5RankerStage2
from src.mi_systems.rewriter import T5Rewriter, RocchioFeedbackRewriter, FirstPassT5Rewriter, T5FeedbackRewriter, RM3FeedbackRewriter, QuReTeCRewriter
from src.mi_systems.response_generator import BARTResponseGenerator
from src.mi_systems.askCQ import SemanticMatchingAskCQ
from src.mi_systems.process_answer import AppendAnswerProcessor

from pathlib import Path
import json
from tqdm import tqdm
import shelve
import torch

torch.cuda.empty_cache()


run_name = "T5+BM25+MonoT5+BART+FeedbackMonoT5-v2-1000"
base_path = "data/generated_files/utterance-picker"
transcripts_path = f"{base_path}/{run_name}/transcripts"

data_generator = CAsTY4DataGenerator(
    dataset_path="data/cast/year_4/annotated_topics.json",
    relevance_judgements_path="data/cast/year_4/cast2022.qrel"
)
pipeline = Pipeline([
    # FirstPassT5Rewriter(),
    # RocchioFeedbackRewriter(),
    # QuReTeCRewriter(),
    T5Rewriter(),
    # T5FeedbackRewriter(),
    SparseRetriever(
        collection="../data/cast_y4_files/trecweb_index/",
        collection_type="trecweb"),
    T5Ranker(),
    # CrossEncoderFeedbackRankerv2(),
    BARTResponseGenerator(),
    # GPT3FeedbackProvider(),
    RandomFeedbackProvider(),
    # T5FeedbackRanker(),
    # CrossEncoderFeedbackRanker()
    T5RankerStage2(),
    # T5FeedbackRanker()
])


Path(transcripts_path).mkdir(parents=True, exist_ok=True)
updated_conversational_turns = []

for conversational_turn in tqdm(data_generator.get_turn(), total=205):
    conversational_turn = pipeline(conversational_turn)
    updated_conversational_turns.append(conversational_turn)
    turn_transcript = []
    for turn in conversational_turn.conversation_history:
        turn_transcript.append({
            "participant": turn["participant"],
            "utterance": turn["utterance"],
            "type": turn["utterance_type"]
        })
    turn_transcript.extend([
        {"participant": "System", "utterance": conversational_turn.system_response,
            "type": conversational_turn.system_response_type},
        {"participant": "User", "utterance": conversational_turn.user_utterance,
            "type": conversational_turn.user_utterance_type}
    ])

    with open(f"{transcripts_path}/{conversational_turn.turn_id}.json", "w") as f:
        json.dump(turn_transcript, f, indent=4, ensure_ascii=False)

    with open(f"{base_path}/{run_name}/{run_name}.run", "a") as f:
        for index, document in enumerate(conversational_turn.ranking):
            f.write(
                f"{conversational_turn.turn_id}\tQ0\t{document.doc_id}\t{index+1}\t{document.score}\t{run_name}\n")

with shelve.open(f'{base_path}/runs') as db:
    db[run_name] = updated_conversational_turns
