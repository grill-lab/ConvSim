from utils import CuriosityDialogsFormatter

dataset_config = [
    {
        "name" : "curiosity_dialogs",
        "data_formatter": CuriosityDialogsFormatter,
        "output_path": "/shared/training_data/curiosity_dialogs_data.jsonl",
        "source" : "https://obj.umiacs.umd.edu/curiosity/curiosity_dialogs.json"
    }
]