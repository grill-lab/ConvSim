from src.base_module.AbstractModule import Pipeline
from src.data_generators import CAsTY4DataGenerator
from src.data_classes.conversational_turn import Document
from src.simulator.answerCQ import GPT3AnswerCQ

data_generator = CAsTY4DataGenerator()
pipeline = Pipeline([GPT3AnswerCQ()])

for conversational_turn in data_generator.get_turn():
    # dummy information need and CQ, should be parsed from data generator
    conversational_turn.information_need = "Events that happened in COP26."
    conversational_turn.system_response = "What about it do you want to know?"

    conversational_turn = pipeline(conversational_turn)
    # dummy documents to evaluate
    conversational_turn.ranking = [
        Document("MARCO_16_3117875026-1", "test", 1.0), 
        Document("KILT_14345195-1", "test", 0.9), 
        Document("KILT_2043296-13", "test", 0.8)
    ]
    score = conversational_turn.evaluate_turn()
    break
