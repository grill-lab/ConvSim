from src.data_generators import CAsTY4DataGenerator
from src.simulator.answerCQ import GPT3AnswerCQ
from src.base_transformer.AbstractTransformer import Pipeline

data_generator = CAsTY4DataGenerator()
pipeline = Pipeline({
    GPT3AnswerCQ()
})

for conversational_turn in data_generator.get_turn():
    # dummy information need and CQ, should be parsed from data generator
    conversational_turn.information_need = "Events that happened in COP26."
    conversational_turn.system_response = "What about it do you want to know?"

    conversational_turn = pipeline.step(conversational_turn)
    print(conversational_turn.user_utterance)
    break