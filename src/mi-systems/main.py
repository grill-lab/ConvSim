import argparse
import pandas as pd

from IPython import embed
from retriever.AbstractRetriever import DummyRetriever
from cn_prediction.AbstractCNPrediction import DummyCNPrediction
from askCQ.AbstractAskCQ import DummySelectCQ
from process_answer.AbstractAnswerProcessor import DummyAnswerProcessor 

# TODO: think about how best to skip certain steps if we don't need them
def main(args):
    # TODO: consider imports like this
    # module = __import__("retriever."+args.retriever)
    # class_ = getattr(module, args.retriever)
    # retriever = class_()
    if args.retriever:
        retriever = DummyRetriever(collection=["Doc1", "Doc2", "document 3"])

    if args.reranker:
        reranker = AbstractReranker()

    if args.clarification_need_prediction:
        cn_prediction = DummyCNPrediction()

    if args.ask_clarifying_question:
        askCQ = DummySelectCQ(question_pool=["Are you interested in this?"])

    if args.answer_processing:
        answer_processor = DummyAnswerProcessor()

    # full pipeline? doesn't make sense like this tbh 
    queries = ["Query 1", "Query 23", "querrrii 3"]
    for query in queries:
        print("###"*20)
        print(f"Processing query: {query}")
        if args.retriever:
            ranking = retriever.retrieve(query)
            print(f"Ranked list of documents: {ranking}")

        if args.reranker:
            ranking = reranker.rerank(query, ranking)

        if args.clarification_need_prediction:
            ask_or_not = cn_prediction.predict_cn(query)
            print(f"Should I ask clarifying question? {ask_or_not}")

        if ask_or_not and args.ask_clarifying_question:
            question = askCQ.ask_cq(query)
            print(f"Asking clarifyig question: {question}")
            # TODO: here we ask USi2.0
            answer = "Yes, that's what I'm looking for."
            new_query = answer_processor.process_answer(question, answer)
            print(f"Processed answer. New query: {new_query}")

            # perform retrieval again or whatever.


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    # perhaps we"ll need separate flags for initial retrieval and for reranking
    parser.add_argument("--retriever", default="AbstractRetriever", type=str,
                            help="Retriever module name.")
    parser.add_argument("--reranker", default="", type=str,
                            help="Reranker module name.")
    parser.add_argument("--clarification_need_prediction",
                         default="DummyCNPrediction", type=str,
                        help="Module name for clarification need prediction.")
    parser.add_argument("--ask_clarifying_question",
                        default="AbstractAskClarifyingQuestion", type=str,
                        help="Module name for asking clarifying questions.")
    parser.add_argument("--answer_processing", default="AbstractProcessAnswer",
                        type=str, help="Module name for processing answers to \
                            clarifying questions.")
    args = parser.parse_args()

    main(args)

