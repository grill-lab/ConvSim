# %% [markdown]
# ## Runs Analysis

# %%
import ir_measures
from ir_measures import *
from pyserini.search.lucene import LuceneSearcher
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from bs4 import BeautifulSoup as bs
import json
import tqdm

# %%
qrels = list(ir_measures.read_trec_qrels('../../data/datasets/cast/year_4/cast2022.qrel'))
searcher = LuceneSearcher('../../data/indexes/sparse/cast/trecweb_index/')
reranker = MonoT5()

MRR = ir_measures.RR(rel=2)
RECALL = ir_measures.R(rel=2)@1000
NDCG_CUT_3 = ir_measures.nDCG@3
MAP = ir_measures.AP(rel=2)@1000

with open('../../data/datasets/cast/year_4/annotated_topics.json', 'r') as f:
    annotated_topics = json.load(f)

# Load gold cqs and rewrites.
with open(
    '../../data/datasets/cast/year_4/gold_cqs_and_rewrites.json', 'r') as f:
    gold_cqs_and_rewrites = json.load(f)

# %%
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


def evaluate_turn(qrels: list, passages: list) -> dict:
    return ir_measures.calc_aggregate(
        [
            MRR,
            RECALL,
            NDCG_CUT_3, 
            MAP
        ], 
        qrels, passages)

# %%
# CAsT Evaluation.
all_documents = []
for topic in tqdm.tqdm(annotated_topics, total=len(annotated_topics)):
    for turn in topic['turn']:
        manual_query = turn['manual_rewritten_utterance']
        turn_id = f"{topic['number']}_{turn['number']}"
        documents = retrieve_passages(manual_query, turn_id, rerank=True)
        all_documents += documents

cast_evaluation = evaluate_turn(qrels, all_documents)
print("CAsT Evaluation: ", cast_evaluation)

# %%
all_documents = []
for turn_id, data in tqdm.tqdm(gold_cqs_and_rewrites.items(), total=len(gold_cqs_and_rewrites)):
    query = data['rewrite']
    all_documents += retrieve_passages(query, turn_id, rerank=True)

rewrite_evaluation = evaluate_turn(qrels, all_documents)
print("Rewrite Evaluation: ", rewrite_evaluation)


