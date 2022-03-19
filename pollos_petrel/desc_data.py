import pandas as pd


def read_training_dataset() -> pd.DataFrame:
    training_dataset_path = "train.csv"
    training_dataset = pd.read_csv(training_dataset_path)
    return training_dataset


trainDF = read_training_dataset()

trainDF.describe().to_csv("describe_train.csv")
