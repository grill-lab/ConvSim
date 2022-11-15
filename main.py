from usi.data_generators import CAsTY4DataGenerator

data_generator = CAsTY4DataGenerator()

for conversational_turn in data_generator.get_turn():
    print(conversational_turn.turn_id)
