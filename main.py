from src.base_module.AbstractModule import Pipeline
from src.data_generators import CAsTY4DataGenerator
from src.data_classes.conversational_turn import Document
from src.simulator.answerCQ import GPT3AnswerCQ
from src.simulator.provide_feedback import GPT3FeedbackProvider
from src.mi_systems.retriever import SparseRetriever
from src.mi_systems.reranker import T5Ranker
from src.mi_systems.rewriter import T5Rewriter
from src.mi_systems.response_generator import BARTResponseGenerator

data_generator = CAsTY4DataGenerator()
pipeline = Pipeline([
    T5Rewriter(),
    SparseRetriever("data/cast/year_4/indexes/content/files/index/sparse"),
    T5Ranker(),
    BARTResponseGenerator(),
    GPT3FeedbackProvider()
])

for conversational_turn in data_generator.get_turn():
    print(conversational_turn.user_utterance)
    print(conversational_turn.rewritten_utterance)

    conversational_turn = pipeline(conversational_turn)
    print(conversational_turn.system_response)
    print(conversational_turn.user_utterance)
    print(conversational_turn.rewritten_utterance)
    break
