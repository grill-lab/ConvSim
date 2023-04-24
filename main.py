from src.base_module.Pipelines import Pipeline, RecursivePipeline
from src.data_generators import CAsTY4DataGenerator
from src.data_classes.conversational_turn import Document
from src.simulator.answerCQ import GPT3AnswerCQ
from src.simulator.provide_feedback import GPT3FeedbackProvider
from src.mi_systems.retriever import SparseRetriever
from src.mi_systems.reranker import T5Ranker
from src.mi_systems.rewriter import T5Rewriter
from src.mi_systems.response_generator import BARTResponseGenerator
from src.mi_systems.askCQ import SemanticMatchingAskCQ
from src.mi_systems.process_answer import AppendAnswerProcessor

from pathlib import Path
import json
from tqdm import tqdm


run_name = "semantic_cq"
base_path = "data/generated_files"
output_path = f"{base_path}/{run_name}/transcripts"

data_generator = CAsTY4DataGenerator(
    dataset_path="data/cast/year_4/annotated_topics.json",
    relevance_judgements_path="data/cast/year_4/cast2022.qrel"
)
pipeline = RecursivePipeline([
    T5Rewriter(),
    SemanticMatchingAskCQ(
        "data/cast/year_4/2022_mixed_initiative_question_pool.json"),
    GPT3AnswerCQ(),
    AppendAnswerProcessor(),
    SparseRetriever(
        collection="../data/cast/year_4/indexes/",
        collection_type="json"),
    T5Ranker(),
    BARTResponseGenerator(),
    GPT3FeedbackProvider()
])


Path(output_path).mkdir(parents=True, exist_ok=True)

for conversational_turn in tqdm(data_generator.get_turn(), total=205):
    conversational_turn = pipeline(conversational_turn)
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

    with open(f"{output_path}/{conversational_turn.turn_id}.json", "w") as f:
        json.dump(turn_transcript, f, indent=4, ensure_ascii=False)

    with open(f"{base_path}/{run_name}.run", "a") as f:
        for index, document in enumerate(conversational_turn.ranking):
            f.write(
                f"{conversational_turn.turn_id}\tQ0\t{document.doc_id}\t{index+1}\t{document.score}\t{run_name}\n")
