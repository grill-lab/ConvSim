from .abstract_simulated_user import AbstractSimulatedUser
from simplet5 import SimpleT5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import random
from sentence_transformers import SentenceTransformer, util

class SimulatedUser(AbstractSimulatedUser):

    def __init__(self, topic: str, facets: list, prior_knowledge: list, patience: int, model_path: str) -> None:
        super().__init__(topic, facets, prior_knowledge, patience)

        # question generator
        self.question_generator = SimpleT5()
        self.question_generator.load_model(
            "t5", model_path, use_gpu=True
        )

        # response evaluator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "castorini/monot5-base-msmarco-10k"
        self.response_evaluator_tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="/shared/models/response_evaluator"
        )
        self.response_evaluator = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, return_dict=True, cache_dir="/shared/models/response_evaluator"
        ).to(self.device).eval()

        # feedback generator
        self.feedback_generator = pipeline(
            'text-generation', 
            model="EleutherAI/gpt-neo-1.3B",
            model_kwargs = {"cache_dir": "/shared/models/feedback_generator"}, 
            device = -1 if self.device == "cpu" else 0
        )

        # sentence similarities
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/shared/models/sentence_embedder')
    
    def __evaluate_response(self, conversation_history: list) -> bool:
        system_response = conversation_history[-1]['system_response']
        last_turn_user_utterance = conversation_history[-1]['user_utterance']
        conversation_text = ''
        for turn in conversation_history[:-1]:
            conversation_text += f"{turn['user_utterance']} {turn['system_response']}"
        conversation_text += f" {last_turn_user_utterance}"
        
        input_ids = self.response_evaluator_tokenizer(
            f"Query: {conversation_text} Document: {system_response} Relevant: ", return_tensors="pt"
        ).to(self.device).input_ids
        outputs = self.response_evaluator.generate(input_ids)
        evaluation = self.response_evaluator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return bool(evaluation)

    
    def generate_utterance(self, conversation_history: list) -> str:

        # start the conversation
        if not conversation_history:
            # model_input = f"Topic: {self.topic} ||| Facets: {self.facets} ||| Knowledge: {self.prior_knowledge} ||| Conversation: "
            model_input = f"Topic: {self.topic} ||| Conversation: "
            utterance = self.question_generator.predict(model_input, num_return_sequences=4, num_beams=4)[0]
            return utterance
        
        # check if system response is relevant
        elif self.__evaluate_response(conversation_history):
            conversation_text = [
                f"{turn['user_utterance']} ||| {turn['system_response']}" for turn in conversation_history
            ]
            conversation_text = ' ||| '.join(conversation_text)
            
            model_input = f"Topic: {self.topic} ||| Facets: {self.facets} ||| Knowledge: {self.prior_knowledge} ||| Conversation: {conversation_text}"
            # model_input = f"Topic: {self.topic} ||| Conversation: {conversation_text}"
            candidate_utterances = self.question_generator.predict(
                model_input, num_return_sequences=4, num_beams=4
            )

            previous_utterances = [turn['user_utterance'] for turn in conversation_history]
            prev_utterance_embeddings = self.embedder.encode(previous_utterances, convert_to_tensor=True).to('cuda')
            candidate_utterance_embedding = self.embedder.encode(candidate_utterances, convert_to_tensor=True).to('cuda')

            hits = util.semantic_search(candidate_utterance_embedding, prev_utterance_embeddings , top_k=1)
            
            utterances = []
            for idx, hit in enumerate(hits):
                if hit[0]['score'] < 0.85:
                    utterances.append(candidate_utterances[idx])

            if utterances:
                return random.choice(utterances)
            else:
                return "Thanks, bye for now."

        # Give feedback   
        else:

            self.patience -= 1

            prompt = (
                "A user is talking to an expert system about Nigeria. The system " 
                "gives the user a wrong answer and the user attempts to provide feedback. "
                "Here is the conversation:\n"
                f"User: {conversation_history[-1]['user_utterance']}\n"
                f"System: {conversation_history[-1]['system_response']}\n"
                "User: No,"
            )

            generation: list = self.feedback_generator(
                prompt, do_sample=False, max_length=70, temperature=0.9
            )

            generation: list = generation[0]['generated_text'].split("\n")
            generated_sequence = [item for item in generation if "User:" in item][-1]
            
            return generated_sequence.split(":")[1].strip()

