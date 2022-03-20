import pandas as pd


def read_training_dataset() -> pd.DataFrame:
    return pd.read_csv(f"train.csv")


read_training_dataset().describe().to_csv(f"describe_train.csv")
