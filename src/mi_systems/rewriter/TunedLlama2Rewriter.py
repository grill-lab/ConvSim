from transformers import TextGenerationPipeline
from .Llama2Rewriter import Llama2Rewriter
from src.data_classes.conversational_turn import ConversationalTurn

class TunedLlama2Rewriter(Llama2Rewriter):

    def __init__(self, pipeline: TextGenerationPipeline, adapter_name: str) -> None:
        super().__init__(pipeline)
        self.adapter_name = adapter_name
    
    def rewrite(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        self.rewriter.model.set_adapter(self.adapter_name)
        return super().rewrite(conversational_turn)
    
    # def __parse_output(self, response: str) -> str:
    #     """Parse the output from llama2 to remove the prompt and any other
    #     unnecessary text.
    #     """
        