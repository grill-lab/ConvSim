import pandas as pd
from simpletransformers.t5 import T5Args, T5Model

from AbstractAnswerCQ import AbstractAnswerCQ


class T5AnswerCQ(AbstractAnswerCQ):
    def __init__(
        self, model_name: str = "t5-small", sep_token: str = "|||", **kwargs
    ) -> None:
        """Answering clarifying questions with T5.

        Args:
            model_name: T5 model name to use compatible with HuggingFace.
                Defaults to 't5-small'.
            **kwargs: Additional keyword arguments for T5Model instance.
        """
        self.model_name = model_name
        self.sep_token = sep_token
        # TODO: make these args easily changable
        self.model_args = T5Args(
            overwrite_output_dir=True,
            manual_seed=42,
            use_multiprocessing=True,
            evaluate_during_training=False,
            reprocess_input_data=True,
            train_batch_size=8,
        )
        self.model_args.num_train_epochs = 5
        self.model_args.learning_rate = 5e-5

        self.model = T5Model("t5", model_name, use_cuda=False, **kwargs)

    def answer_cq(self, clarifying_question: str, context: str = None) -> str:
        """Answers given clarifying question based on self.information_need.

        Args:
            clarifying_question: A question to answer.
            context: Context to be used (e.g., query).
        """
        # concat input # TODO: define what it'll look like
        input_str = self._create_input_string(clarifying_question, context)
        return self.model.predict(input_str)

    def answer_cq_batch(self):
        # TODO
        pass

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, **kwargs) -> None:
        """Train the model using the given data.

        Args:
            train_df: Data for training with 'source_text' and 'target_text'.
            eval_df: Data for validation with 'source_text' and 'target_text'.
        """
        # assert if the data is good

        self.model.train(
            train_df=train_df,
            eval_df=eval_df,
            source_max_token_len=512,
            target_max_token_len=128,
            outputdir=f"models/T5AnswerCQ-finetuned-x",
            **kwargs,
        )

    def eval(self, data: pd.DataFrame) -> None:
        """Evaluate the model with the given data.

        Args:
            data: Dataset to evaluate the model on. Contains...
        """
        pass

    def _create_input_str(
        self, inf_need: str, clarifying_question: str, context: str = None
    ) -> str:
        """Create the string fit to be an input to the T5Model.

        Args:
            inf_need: Information need textual description to base answers on.
            clarifying_question: Clarifying question that needs an answer.
            context (optional): String to be added as context (e.g., query).
        """
        if context:
            input_str = inf_need + f" {self.sep_token} " + context
        else:
            input_str = inf_need

        input_str += f" {self.sep_token} " + clarifying_question
        return input_str

    def load_df(self, clariq_path: str) -> pd.DataFrame:
        """Load and organize clariq dataset to be fit for T5 training.

        Args:
            clariq_path: a path to the .csv dataset.
        Returns:
            pd.DataFrame with [prefix, input_text, target_text] columns.
        """
        clariq = pd.read_csv(clariq_path, sep="\t")
        t5_ready = clariq[["facet_desc", "question"]].dropna()
        t5_ready["target_text"] = clariq["answer"]
        t5_ready["prefix"] = "QA"
        t5_ready["input_text"] = t5_ready.apply(
            lambda t: self._create_input_str(t["facet_desc"], t["question"]), 1
        )
        t5_ready = t5_ready[["prefix", "input_text", "target_text"]]
        print("Data shape:", t5_ready.shape)
        return t5_ready


if __name__ == "__main__":
    # init model
    t5a = T5AnswerCQ("t5-small", epochs=5, lr=5e-5)

    # load data
    train_df = T5AnswerCQ.load_df("../../data/clariq/train.tsv")
    eval_df = T5AnswerCQ.load_df("../../data/dev.tsv")

    # train
    t5a.train(eval_df, eval_df)
