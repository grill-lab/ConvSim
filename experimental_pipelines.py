import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers import pipeline as transformers_pipeline

from src.base_module.Pipelines import (MultiActionLlamaAgentPipeline, Pipeline,
                                       RecursivePipeline,
                                       SingleActionLlamaAgentPipeline,
                                       SingleActionPipelineAgent)
from src.mi_systems.askCQ import T5AskCQ
from src.mi_systems.reranker import T5Ranker
from src.mi_systems.response_generator import (BARTResponseGenerator,
                                               T5ResponseGenerator)
from src.mi_systems.retriever import DenseRetriever, SparseRetriever
from src.mi_systems.rewriter import (ComboRewriter, FeedbackRewriter,
                                     Llama2Rewriter, OpenAIFeedbackRewriter,
                                     QuReTeCRewriter, T5FeedbackRewriterv2,
                                     T5Rewriter, GPT3TurboRewriterAndEditor)
from src.simulator.provide_feedback import (HumanFeedbackProvider,
                                            OpenAIFeedbackProvider,
                                            PromptedLlama2FeedbackProvider,
                                            TunedLlama2FeedbackProvider)


def standard_baseline() -> Pipeline:
    """This is a baseline run. It uses a BM25 retriever, a T5 rewriter, a T5
    ranker and a BART summariser.
    
    Notes: We retrieve 1000 documents, and rerank all of them using the T5 
    ranker."""
    return Pipeline([
        T5Rewriter(),
        SparseRetriever(
            collection="../../data/ikat/data/indexes/sparse",
            collection_type="json"),
        T5Ranker(),
        T5ResponseGenerator(),
    ])


def standard_baseline_with_openai_simulator() -> Pipeline:
    """See above, but with an OpenAI simulator for one round of feedback."""
    return RecursivePipeline([
        T5Rewriter(),
        SparseRetriever(
            collection="../../data/ikat/data/indexes/sparse",
            collection_type="json"),
        T5Ranker(),
        T5ResponseGenerator(),
        OpenAIFeedbackProvider()
    ])


def rewrite_then_edit_baseline() -> Pipeline:
    """Uses the Rewrite then edit approach based on GPT3."""
    return Pipeline([
        GPT3TurboRewriterAndEditor(),
        SparseRetriever(
            collection="../../data/ikat/data/indexes/sparse",
            collection_type="json"),
        T5Ranker(),
        T5ResponseGenerator(),
    ])


def rewrite_then_edit_baseline_with_openai_simulator() -> Pipeline:
    """See above but with OpenAI simulator for one round of feedback."""
    return RecursivePipeline([
        GPT3TurboRewriterAndEditor(),
        SparseRetriever(
            collection="../../data/ikat/data/indexes/sparse",
            collection_type="json"),
        T5Ranker(),
        T5ResponseGenerator(),
        OpenAIFeedbackProvider()
    ])

# def load_model_with_adapters(base_model_id: str,
#                              adapter_dict: dict[str, str] = None):
#     """Creates a base llama model with the correct quantization settings.
#     Also returns the tokenizer."""
    
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=getattr(torch, "float16"),
#         bnb_4bit_use_double_quant=False,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_id,
#         quantization_config=quant_config,
#         device_map="auto",
#         trust_remote_code=True
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_id,
#         add_bos_token=True, trust_remote_code=True)

#     if adapter_dict:
#         for adapter_name, adapter_path in adapter_dict.items():
#             model = PeftModel.from_pretrained(model=model, 
#                                             model_id=adapter_path,
#                                             adapter_name=adapter_name,
#                                             offload_folder="offload")
#         # This basically creates a llama model and not a Peft Model.
#         # model = model.merge_and_unload()
    
#     return model, tokenizer


# def standard_baseline_gpt3_rewriter() -> Pipeline:
#     """GPT3 baseline run. It uses a BM25 retriever, a T5 rewriter, a T5
#     ranker and a BART summariser.
    
#     Notes: We retrieve 1000 documents, and rerank all of them using the T5 
#     ranker."""
#     return Pipeline([
#         OpenAIFeedbackRewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#     ])

# def manual_baseline() -> Pipeline:
#     """Baseline run but with manual queries."""
#     return Pipeline([
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#     ])


# def tuned_simulation_baseline() -> Pipeline:
#     """
#     This is a simulation baseline run. It uses a BM25 retriever, a T5 rewriter,
#     a T5 ranker and a BART summariser with a tuned LLAMA2 feedback provider for
#     some rounds of feedback.
    
#     Notes: We retrieve 1000 documents, and rerank all of them using the T5
#     ranker.
#     """
#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         T5Rewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#         TunedLlama2FeedbackProvider(
#             pipeline=inference_pipeline,
#             adapter_name="simulator")
#     ], max_feedback_rounds=3, min_ndcg=1.00)


# def tuned_simulation_baseline_with_rm3_rewriter() -> Pipeline:
#     """
#     Same as above but using rm3 for query rewriting.
#     """
#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         FeedbackRewriter(
#             collection="../data/indexes/sparse/ikat",
#             feedback_type='rm3', #rocchio
#         ),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         T5ResponseGenerator(),
#         TunedLlama2FeedbackProvider(
#             pipeline=inference_pipeline,
#             adapter_name="simulator")
#     ], max_feedback_rounds=1, min_ndcg=1.00)


# def tuned_simulation_baseline_with_gpt3_rewriter() -> Pipeline:
#     """
#     Using OpenAI GPT-3 for query rewriting.
#     """
#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         OpenAIFeedbackRewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         T5ResponseGenerator(),
#         TunedLlama2FeedbackProvider(
#             pipeline=inference_pipeline,
#             adapter_name="simulator"),
#     ], max_feedback_rounds=1, min_ndcg=1.00)

# def tuned_simulation_baseline_with_distilled_t5_rewriter() -> Pipeline:
#     """
#     Using distilled T5 for query rewriting.
#     """
#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         T5FeedbackRewriterv2(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         T5ResponseGenerator(),
#         TunedLlama2FeedbackProvider(
#             pipeline=inference_pipeline,
#             adapter_name="simulator"),
#     ], max_feedback_rounds=1, min_ndcg=1.00)

# def tuned_simulation_baseline_with_combo_rewriter() -> Pipeline:
#     """
#     Combines T5 canard with distilled t5 for query rewriting.
#     """
#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         ComboRewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         T5ResponseGenerator(),
#         TunedLlama2FeedbackProvider(
#             pipeline=inference_pipeline,
#             adapter_name="simulator"),
#     ], max_feedback_rounds=1, min_ndcg=1.00)

# def prompted_simulation_baseline() -> Pipeline:
#     """This is another simulation baseline run. It uses a BM25 retriever, a T5
#     rewriter, a T5 ranker and a BART summariser with a prompted LLAMA2 feedback
#     provider for some rounds of feedback.
    
#     Notes: We retrieve 1000 documents, and rerank all of them using the T5
#     ranker.
#     """
#     llama, llama_tokenizer = load_model_with_adapters(
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     return RecursivePipeline([
#         T5Rewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#         PromptedLlama2FeedbackProvider(
#             pipeline=transformers_pipeline(
#                 task="text-generation",
#                 model=llama,
#                 tokenizer=llama_tokenizer
#             ))
#     ], max_feedback_rounds=1, min_ndcg=1.00)

# def gpt3_simulation_baseline() -> Pipeline:
#     """Simulation run using GPT-3 for feedback.
    
#     Notes: We retrieve 1000 documents, and rerank all of them using the T5
#     ranker.
#     """

#     return RecursivePipeline([
#         T5Rewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         T5ResponseGenerator(),
#         OpenAIFeedbackProvider(),
#     ], max_feedback_rounds=1, min_ndcg=1.00)



# def human_feedback_baseline() -> Pipeline:
#     """This is a human feedback baseline run. It uses a BM25 retriever, a T5
#     rewriter, a T5 ranker and a BART summariser with a human feedback provider
#     for some rounds of feedback.
    
#     Notes: We retrieve 1000 documents, and rerank all of them using the T5
#     ranker.
#     """
#     return RecursivePipeline([
#         T5Rewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#         HumanFeedbackProvider()
#     ], max_feedback_rounds=1, min_ndcg=1.00)

# def feedback_reranker() -> Pipeline:

#     """This run uses the same components as the simulation baseline, but we
#     we collect user feedback after the initial retrieval pass, then use that
#     feedback to rerank the documents. After that, we generate another system
#     response."""

#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     return Pipeline([
#         T5Rewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#         TunedLlama2FeedbackProvider(
#             model=llama,
#             tokenizer=llama_tokenizer, 
#             adapter_name="simulator"),
#         T5Rewriter(),
#         T5Ranker(),
#         BARTResponseGenerator()
#     ])


# def dense_baseline() -> Pipeline:
#     """# This run is a dense retrieval baseline run with a colbert encoder."""

#     return Pipeline([
#         DenseRetriever(
#             dense_shards=[
#                 "../data/indexes/dense/ikat/ikat_2023_passages_00",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_01",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_02",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_03",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_04",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_05",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_06",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_07",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_08",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_09",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_10",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_11",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_12",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_13",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_14",
#                 "../data/indexes/dense/ikat/ikat_2023_passages_15",
#             ],
#             collection="../data/indexes/sparse/ikat",
#         ),
#         BARTResponseGenerator()
#     ])


# def single_action_agent() -> SingleActionLlamaAgentPipeline:
#     """This run uses a single action agent model with a BM25 retriever, a T5
#     rewriter, a T5 ranker and a BART summariser."""

#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#             "agent": "../data/models/llama-7b-agent-tuned-balanced-no-rerank-3-epochs"
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return SingleActionLlamaAgentPipeline(
#         agent_model=inference_pipeline,
#         adapter_name="agent",
#         searcher=SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         rewriter=T5Rewriter(),
#         reranker=T5Ranker(),
#         user_simulator=TunedLlama2FeedbackProvider(
#             pipeline=inference_pipeline,
#             adapter_name="simulator"),
#         response_generator=BARTResponseGenerator()
#     )

# def single_action_agent_with_gpt3() -> SingleActionLlamaAgentPipeline:

#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#             "agent": "../data/models/llama-7b-agent-tuned-balanced-no-rerank-3-epochs"
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return SingleActionLlamaAgentPipeline(
#         agent_model=inference_pipeline,
#         adapter_name="agent",
#         searcher=SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         rewriter=T5Rewriter(),
#         reranker=T5Ranker(),
#         user_simulator=OpenAIFeedbackProvider(),
#         response_generator=T5ResponseGenerator()
#     )

# def multi_action_agent() -> MultiActionLlamaAgentPipeline:
#     """This run uses a multi action agent model with a BM25 retriever, a T5
#     rewriter, a T5 ranker and a BART summariser."""

#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#             "agent": "../data/models/llama-7b-agent-tuned-balanced-no-rerank-3-epochs"
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     inference_pipeline = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return MultiActionLlamaAgentPipeline(
#         agent_model=inference_pipeline,
#         adapter_name="agent",
#         searcher=SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         rewriter=T5Rewriter(),
#         reranker=T5Ranker(),
#         user_simulator=TunedLlama2FeedbackProvider(
#             pipeline=inference_pipeline,
#             adapter_name="simulator"),
#         response_generator=BARTResponseGenerator()
#     )


# def single_action_agent_pipeline_with_feedback_rewriter() -> SingleActionPipelineAgent:

#     agent = transformers_pipeline(
#         "text-classification",
#         model="../data/models/longformer-base-4096-action-classifier/checkpoint-2500",
#         tokenizer="../data/models/longformer-base-4096-action-classifier/checkpoint-2500",
#         device="cuda:3")
    
#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     simulator = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )
    
#     return SingleActionPipelineAgent(
#         agent_model=agent,
#         searcher=SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         # rewriter=T5FeedbackRewriterv2(),
#         rewriter=OpenAIFeedbackRewriter(),
#         reranker=T5Ranker(),
#         clarifier=T5AskCQ(),
#         user_simulator=TunedLlama2FeedbackProvider(
#             pipeline=simulator,
#             adapter_name="simulator"),
#         response_generator=T5ResponseGenerator()
#     )


# def openai_rewriter_and_simulator() -> Pipeline:
#     return RecursivePipeline([
#         OpenAIFeedbackRewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         T5ResponseGenerator(),
#         OpenAIFeedbackProvider()
#     ], max_feedback_rounds=1, min_ndcg=1.00)


# def prompted_test() -> Pipeline:
#     llama, llama_tokenizer = load_model_with_adapters(
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     pipeline=transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         # Llama2Rewriter(pipeline=pipeline),
#         T5Rewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#         PromptedLlama2FeedbackProvider(pipeline=pipeline)
#     ], max_feedback_rounds=1, min_ndcg=1.00)


# def openai_with_prompted_simulator() -> Pipeline:
#     llama, llama_tokenizer = load_model_with_adapters(
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     pipeline=transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         # Llama2Rewriter(pipeline=pipeline),
#         OpenAIFeedbackRewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#         PromptedLlama2FeedbackProvider(pipeline=pipeline)
#     ], max_feedback_rounds=1, min_ndcg=1.00)

# def tuned_llama_rewriter_with_openai_simulator() -> Pipeline:
#     llama, llama_tokenizer = load_model_with_adapters(
#         base_model_id="meta-llama/Llama-2-7b-chat-hf",
#         adapter_dict={
#             "rewriter": "../data/models/llama-7b-rewriter-canard-4-epochs",
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         }
#     )

#     pipeline=transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )

#     return RecursivePipeline([
#         TunedLlama2Rewriter(pipeline=pipeline, adapter_name="rewriter"),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         BARTResponseGenerator(),
#         # OpenAIFeedbackProvider()
#         TunedLlama2FeedbackProvider(pipeline=pipeline, adapter_name="simulator")
#     ], max_feedback_rounds=1, min_ndcg=1.00)


# def canard_data_annotation() -> Pipeline:

#     return RecursivePipeline([
#         OpenAIFeedbackRewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5ResponseGenerator(),
#         OpenAIFeedbackProvider()
#     ], max_feedback_rounds=1, min_ndcg=1.00)


# def quretec_with_tuned_llama_baseline() -> Pipeline:
#     llama, llama_tokenizer = load_model_with_adapters(
#         adapter_dict={
#             "simulator": "../data/models/llama-7b-simulator-tuned",
#         },
#         base_model_id="meta-llama/Llama-2-7b-chat-hf"
#     )

#     simulator = transformers_pipeline(
#         task="text-generation",
#         model=llama,
#         tokenizer=llama_tokenizer
#     )
#     return RecursivePipeline([
#         QuReTeCRewriter(),
#         SparseRetriever(
#             collection="../data/indexes/sparse/ikat",
#             collection_type="trecweb"),
#         T5Ranker(),
#         T5ResponseGenerator(),
#         TunedLlama2FeedbackProvider(
#             pipeline=simulator,
#             adapter_name="simulator")
#     ], max_feedback_rounds=1, min_ndcg=1.00)