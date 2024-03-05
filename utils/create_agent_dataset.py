# %%
import json
import ir_measures
from pyserini.search.lucene import LuceneSearcher
from bs4 import BeautifulSoup as bs
import tqdm
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential
import random
import tiktoken

# %%
dataset = []
# Load the base data.
with open('../../data/datasets/cast/year_4/annotated_topics.json', 'r') as f:
    annotated_topics = json.load(f)

# Load gold cqs and rewrites.
with open(
    '../../data/datasets/cast/year_4/gold_cqs_and_rewrites.json', 'r') as f:
    gold_cqs_and_rewrites = json.load(f)

# Load the qrels.
qrels = list(ir_measures.read_trec_qrels(
    '../../data/datasets/cast/year_4/cast2022.qrel'))

# load the searcher.
searcher = LuceneSearcher('../../data/indexes/sparse/cast/trecweb_index')
reranker = MonoT5()

# metrics
MRR = ir_measures.RR(rel=2)
RECALL = ir_measures.R(rel=2)@1000
NDCG_CUT_3 = ir_measures.nDCG@3
MAP = ir_measures.AP(rel=2)@1000

load_dotenv()
OPENAI_CLIENT = OpenAI()
ENCODER = tiktoken.get_encoding("cl100k_base")

# %%
def get_text(doc_id: str):
    """Returns the passage text for a given doc_id."""
    doc_id, passage_id = doc_id.rsplit("-", 1)
    document = searcher.doc(doc_id).raw()
    document = bs(document, "lxml")
    passages = document.find_all("passage")
    for idx, passage in enumerate(passages):
        if idx == int(passage_id):
            return passage.text.strip()

def create_sample(utterances: list, top_passages: list = list(), 
                  action: str = None, system_utterance: str = None,
                  turn_id: str = None):
    """Creates a sample for the dataset."""
    sample = {
        "utterances": utterances,
        "top_passages": top_passages,
        "action": action,
        "system_utterance": system_utterance,
        "turn_id": turn_id
    }
    return sample

def evaluate_turn(qrels: list, passages: list) -> dict:
    return ir_measures.calc_aggregate(
        [
            MRR,
            RECALL,
            NDCG_CUT_3, 
            MAP
        ], 
        qrels, passages)

def convert_hits_to_scoredDoc(hits: list, turn_id: str) -> list:
    """Converts a list of hits to a list of scoredDoc."""
    return [
        ir_measures.ScoredDoc(turn_id, hit.docid, hit.score) if hasattr(hit, 'docid') 
        else ir_measures.ScoredDoc(turn_id, hit.metadata['docid'], hit.score) for hit in hits
    ]

def retrieve_passages(query: str, query_id: str, num_results: int = 1000, rerank: bool=True):
    """Retrieves passages for a given query."""
    documents = searcher.search(query, num_results)
    passages = [Text(p.text.strip(), {'docid': f"{document.docid}-{p['id']}"}, 0) 
                for document in documents 
                for p in bs(document.raw, "lxml").find_all("passage")]
    passages = passages[:1000]

    if rerank:
        passages = reranker.rerank(Query(query), passages)

    scored_docs = convert_hits_to_scoredDoc(passages, query_id)

    return scored_docs

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(ENCODER.encode(string))


@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def get_message(conversation_str: str, system_prompt: str):
    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": conversation_str
            }
        ],
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# %%
from string import Template
# load stashes
# response_stash = {}
# retrieval_stash = {}
with open('../../data/datasets/agent_training/retrieval_stash.json', 'r') as f:
    retrieval_stash = json.load(f)
    for turn in retrieval_stash:
        for key in retrieval_stash[turn]:
            retrieval_stash[turn][key] = [ir_measures.ScoredDoc(*item) for item in retrieval_stash[turn][key]]

with open('../../data/datasets/agent_training/response_stash.json', 'r') as f:
    response_stash = json.load(f)

summary_prompt = "You will receive three passages. Your task is to create a short and concise summary of all passages to answer the following: $query There is no need to reference the passages in your response."

def generate_samples(turn_id, utterances):
    # Sample for CQS and rewrite
    qrels_for_turn = [qrel for qrel in qrels if qrel.query_id == turn_id]

    rewrite = gold_cqs_and_rewrites.get(turn_id).get("rewrite")
    rewrite_results = retrieval_stash[turn_id]['rewrite']

    # rewrite_results = retrieve_passages(rewrite, turn_id)
    rewrite_scores = evaluate_turn(qrels_for_turn, rewrite_results)
    
    # raw_query = utterances[-1]['User']
    # raw_results = retrieval_stash[turn_id]['raw']
    # raw_results = retrieve_passages(raw_query, turn_id, rerank=False)

    # retrieval_stash[turn_id] = {"rewrite": rewrite_results, "raw": raw_results}

    # Sample for CQ and search.
    if rewrite_scores[MRR] > 0.3 and rewrite_scores[RECALL] > 0.7 and rewrite_scores[NDCG_CUT_3] > 0.25:
        sample = create_sample(utterances, [], "search", rewrite, turn_id)
        dataset.append(sample)
    else:
        clarifying_question = gold_cqs_and_rewrites.get(turn_id).get("cq")
        sample = create_sample(
            utterances, [], "clarify", clarifying_question, turn_id)
        dataset.append(sample)
    
    # Get relevant passage texts.
    # relevant_passages = [
    #     get_text(qrel.doc_id) for qrel in qrels_for_turn if qrel.relevance > 2 and get_text(qrel.doc_id)]
    # if len(relevant_passages) == 0:
    #     print(f"No relevant passages found for turn {turn_id}.")
    #     return
    
    # relevant_passages = random.sample(relevant_passages, min(3, len(relevant_passages)))
    # passage_str = "\n".join(f"({idx}): {rp}" for idx, rp in enumerate(relevant_passages, start=1))

    # while num_tokens_from_string(summary_prompt + "" + passage_str) > 3500:
    #     relevant_passages = relevant_passages[:-1]
    #     passage_str = "\n".join(f"({idx}): {rp}" for idx, rp in enumerate(relevant_passages, start=1))
    

    # response = get_message(
    #     passage_str, Template(summary_prompt).substitute(query=rewrite))
    
    # response_stash[turn_id] = {'passages': relevant_passages, 'response': response}
    
    if turn_id in response_stash:
        relevant_passages = response_stash[turn_id]['passages']
        response = response_stash[turn_id]['response']
        
        sample = create_sample(
            utterances, relevant_passages, "respond", response, turn_id)
        dataset.append(sample)

# %%
def process_base_data():
    parsed_turns = set()
    for topic in tqdm.tqdm(annotated_topics):
        for index, turn in enumerate(topic['turn']):
            turn_id = f"{topic['number']}_{turn['number']}"
            if turn_id in parsed_turns:
                continue
            parsed_turns.add(turn_id)

            # Store all the utterances.
            utterances = []
            for previous_turn in topic['turn'][:index]:
                utterances.append({
                    'User': previous_turn.get("utterance"),
                    'System': previous_turn.get("response")
                })
            utterances.append({'User': turn.get("utterance")})
            generate_samples(turn_id, utterances)
        

# %%
process_base_data()
# Save stashes
# with open('../../data/datasets/agent_training/retrieval_stash.json', 'w') as f:
#     json.dump(retrieval_stash, f, indent=4, ensure_ascii=False)

# with open('../../data/datasets/agent_training/response_stash.json', 'w') as f:
#     json.dump(response_stash, f, indent=4, ensure_ascii=False)

# %%
import pathlib
import re
import hashlib
# Parse generated data.
def process_transcript_data(
        path: str = '../../data/transcripts/convsim_outputs'
    ):
    """Loads the conversations for training.
    
    Args:
        path: Path to the conversations shelve db.
    """
    seen_conversations = set()

    for directory in tqdm.tqdm(pathlib.Path(path).iterdir()):
        for subdirectory in directory.iterdir():
            if not subdirectory.is_dir():
                continue
            for file in subdirectory.iterdir():
                # Get the file basename
                basename = file.name.split('.')[0]
                match = re.match(r'^\d+_\d+-\d+$', basename)
                if match:
                    basename = match.string
                else:
                    continue

                with open(file, 'r') as f:
                    try:
                        conversation = json.load(f)
                    except json.decoder.JSONDecodeError:
                        continue
                # Count feedback turns.
                feedback_turns = [
                    turn for turn in conversation if turn['type'] == 'feedback']
                long_feedback_turns = False
                # Check the content of the feedback turns
                for turn in feedback_turns:
                    if len(turn['utterance'].split()) > 25:
                        long_feedback_turns = True
                    
                if len(feedback_turns) > 2 or len(feedback_turns) == 0 or long_feedback_turns:
                    continue

                # Get the hash of the conversation
                conversation_hash = hashlib.sha256(
                    json.dumps(conversation).encode('utf-8')).hexdigest()
                
                if conversation_hash in seen_conversations:
                    continue
                seen_conversations.add(conversation_hash)

                # Parse conversations.
                utterances = []
                current_turn = {}
                for turn in conversation:
                    if turn['participant'] == "User":
                        current_turn['User'] = turn['utterance']
                    else:
                        current_turn['System'] = turn['utterance']
                        utterances.append(current_turn)
                        current_turn = {}
                if current_turn:
                    utterances.append(current_turn)
                
                qrels_for_turn = [
                    qrel for qrel in qrels if qrel.query_id == basename]
                raw_results = retrieval_stash[basename]['raw']


                rewrite_results = retrieval_stash[basename]['rewrite']
                rewrite_scores = evaluate_turn(qrels_for_turn, rewrite_results)
                rewrite = gold_cqs_and_rewrites.get(basename).get("rewrite")

                bad_passages = [get_text(doc.doc_id) for doc in raw_results[:3]]
                if rewrite_scores[MRR] > 0.3 and rewrite_scores[RECALL] >= 0.7 and rewrite_scores[NDCG_CUT_3] >= 0.25:
                    sample = create_sample(
                        utterances, bad_passages, "search", rewrite, basename)
                    dataset.append(sample)
                else:
                    clarifying_question = gold_cqs_and_rewrites.get(basename).get("cq")
                    sample = create_sample(
                        utterances, bad_passages, "clarify", clarifying_question, basename)
                    dataset.append(sample)
                
                if rewrite_scores[RECALL] >= 0.7 and rewrite_scores[MRR] <= 0.3:
                    sample = create_sample(
                        utterances, bad_passages, "rerank", rewrite, basename)
                    dataset.append(sample)
                
                # Get relevant passage texts.
                try:
                    response = response_stash[basename]['response']
                    relevant_passages = response_stash[basename]['passages']
                        
                    sample = create_sample(
                        utterances, relevant_passages, "respond", response, basename)
                    dataset.append(sample)      
                except:
                    continue

# %%
process_transcript_data()

# %%
with open('../../data/datasets/agent_training/agent_dataset.jsonl', 'w') as f:
    for sample in dataset:
        json.dump(sample, f)
        f.write('\n')

