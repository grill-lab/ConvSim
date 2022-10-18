import argparse
import pandas as pd

def main(args):
    module = __import__(module_name)
    class_ = getattr(module, class_name)
    instance = class_()


if __name__=="__main__":

    parser = argparse..ArgumentParser(description="Process some integers.")
    # perhaps we"ll need separate flags for initial retrieval and for reranking
    parser.add_argument("--retrieval", default="AbstractRetrieval", type=str,
                            help="Retrieval module name.")
    parser.add_argument("--clarification_need_prediction", default="", type=str,
                            help="Module name for clarification need prediction.")
    parser.add_argument("--ask_clarifying_question", default="AbstractAskClarifyingQuestion", type=str,
                            help="Module name for asking clarifying questions.")
    parser.add_argument("--answer_processing", default="AbstractProcessAnswer", type=str,
                            help="Module name for processing answers to
                            clarifying questions.")
    args = parser.parse_args()

    main(args)

