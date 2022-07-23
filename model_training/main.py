from simplet5 import SimpleT5
from config import dataset_config
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    for data in dataset_config:

        data_formatter = data["data_formatter"]()
        data_source = data["source"]

        dataset = data_formatter.convert_to_dataframe(data_source)
        train_df, test_df = train_test_split(dataset, test_size=0.01)

        model = SimpleT5()
        model.from_pretrained(model_type="t5", model_name="t5-base")
        model.train(
            train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=512, 
            target_max_token_len=128, 
            batch_size=8, 
            max_epochs=20, 
            use_gpu=True,
            outputdir=f"/shared/models/USi-finetuned-{data['name']}"
        )
