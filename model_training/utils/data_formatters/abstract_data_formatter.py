from abc import ABC, abstractclassmethod, abstractmethod
import pandas as pd

class AbstractDataFormatter(ABC):

    @abstractmethod
    def convert_to_dataframe(self, data_path: str) -> pd.DataFrame:
        """
        Takes in a path to a dataset, and converts it to a pandas dataframe
        with `source_text` and `target_text` columns as required by simpleT5

        Args:
            data_path: Path to dataset

        Returns:
            Dataframe with `source_text` and `target_text` columns
        """
        pass