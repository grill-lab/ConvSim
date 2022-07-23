from utils import CuriosityDialogsFormatter, CAsTFormatter

dataset_config = [
    {
        "name" : "cast",
        "data_formatter" : CAsTFormatter,
        "source" : "/shared/training_data/conversations.json"
    },
    # {
    #     "name" : "curiosity_dialogs",
    #     "data_formatter": CuriosityDialogsFormatter,
    #     "source" : "https://obj.umiacs.umd.edu/curiosity/curiosity_dialogs.json"
    # }
]