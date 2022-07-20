from abc import ABC, abstractclassmethod, abstractmethod
from datasets import DatasetDict

class AbstractDataFormatter(ABC):

    @abstractmethod
    def convert_to_jsonlines(self, data_path: str) -> None:
        """
        Takes in a path to a dataset, extract salient information as needed then
        and writes it out to a jsonlines file

        Args:
            data_path: Path to dataset

        Returns:
            None
        """
        pass

    @abstractmethod
    def load_dataset(self, data_path: str) -> DatasetDict:
        """
        Takes in a path to a dataset already formatted in jsonlines, formats
        input to model, and creates train and test splits

        Args:
            data_path: Path to dataset

        Returns:
            A DatasetDict instance with train and test splits for training
        """
        pass