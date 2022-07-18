from .abstract_search_system import AbstractSearchSystem
from typing import Dict, List, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pyserini.search.lucene import LuceneSearcher
import torch
from bs4 import BeautifulSoup as bs
import lxml

prediction_tokens = ['▁false', '▁true']

# num_of_gpus = torch.cuda.device_count()
# print(num_of_gpus)

class CAsTSearchSystem(AbstractSearchSystem):

    def __init__(self, index_path: str) -> None:

        rewriter_model_name: str = "castorini/t5-base-canard"
        ranker_model_name: str = "castorini/monot5-base-msmarco-10k"
        summariser_model_name: str = "facebook/bart-large-cnn"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        rewriter_cache_directory = '/shared/models/rewriter'
        ranker_cache_directory = '/shared/models/ranker'
        summariser_cache_directory = '/shared/models/summariser'

        # initialise models and tokenizers
        self.rewriter, self.rewriter_tokenizer = self.__get_model_and_tokenizer(
            rewriter_model_name, cache_dir = rewriter_cache_directory
        )
        self.ranker, self.ranker_tokenizer = self.__get_model_and_tokenizer(
            ranker_model_name, cache_dir = ranker_cache_directory
        )
        # Summariser will use pipeline API
        self.summariser: pipeline = pipeline(
            "summarization", model=summariser_model_name, 
            model_kwargs = {"cache_dir": summariser_cache_directory},
            device = -1 if self.device == 'cpu' else 0
        )

        # initialise search system (only sparse as dense will require sharding)
        self.searcher: LuceneSearcher = LuceneSearcher(index_path)

        self.token_false_id = self.ranker_tokenizer.get_vocab()[prediction_tokens[0]]
        self.token_true_id  = self.ranker_tokenizer.get_vocab()[prediction_tokens[1]]

    def __get_model_and_tokenizer(self, model_name: str, cache_dir: str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        Retrieves model and tokenizer from huggingface based on input string

        Args:
            model_name: Name of model to retrieve from huggingface.com
            cache_dir: Directory to cache model files for easier retrieval later on

        Returns:
            A model and tokenizer from huggingface.com
        """
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, return_dict=True, cache_dir=cache_dir
        ).to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        return model, tokenizer
    
    def rewrite_query(self, query: str, context: List) -> str:
        # format input for model
        reformated_context = " ||| ".join(context)
        rewriter_input = reformated_context + f" ||| {query}"

        # generate rewrite
        tokenized_context = self.ranker_tokenizer.encode(
            rewriter_input, return_tensors="pt"
        ).to(self.device)
        output_ids = self.rewriter.generate(
            tokenized_context, max_length=200, num_beams=4, 
            repetition_penalty=2.5, 
            early_stopping=True).to(self.device)
        
        rewrite = self.rewriter_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return rewrite
    
    def retrieve_documents(self, query: str) -> List:
        hits = self.searcher.search(query)
        hits = [{'id': hit.docid, 'body': hit.raw} for hit in hits]
        return hits

    def rank_passages(self, query: str, documents: List) -> List:
        # collect passages
        passages: List = []
        for document in documents:
            document_id = document['id']
            processed_document = bs(document['body'], "lxml")
            extracted_passages = processed_document.find_all("passage")
            extracted_passages = [
                {
                    'id': f"{document_id}-{passage['id']}", 
                    "body": passage.text
                } 
                    for passage in extracted_passages
            ]
            passages.extend(extracted_passages)
        
        passages = passages[:100]
        # rank
        input_ids = self.ranker_tokenizer(
            [f"Query: {query} Document: {passage['body']} Relevant: " for passage in passages], 
            return_tensors="pt", padding=True, truncation=True
        ).to(self.device).input_ids

        outputs = self.ranker.generate(
            input_ids,
            return_dict_in_generate=True, 
            output_scores=True 
        )
    
        scores = outputs.scores[0][:, [self.token_false_id, self.token_true_id]]
        scores = torch.nn.functional.softmax(scores, dim=1)
        probabilities = scores[:, 1].tolist()

        for passage, probability in zip(passages, probabilities):
            passage['score'] = probability

        ranked_passages = sorted(passages, key=lambda x: x['score'], reverse=True)
        return ranked_passages
    
    def generate_summary(self, passages: List, threshold: int, **kwargs) -> str:
        # get the top three passages
        top_n_passages = ' '.join([passage['body'] for passage in passages[:threshold]])
        output = self.summariser(
            top_n_passages, 
            max_length=200, 
            min_length=30, 
            do_sample=False,
            **kwargs,
        )
        return output[0]['summary_text']


