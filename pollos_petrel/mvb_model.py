from sklearn.model_selection import train_test_split
from pollos_petrel import read_training_dataset
import pandas as pd


def split_data(dataset: pd.DataFrame) -> pd.DataFrame:
    numeric = dataset.drop(columns=["target", "id"])
    return numeric


def split_target(dataset: pd.DataFrame) -> pd.DataFrame:
    target = dataset[["target"]]
    return target


def preprocces_training_data() -> dict:
    training_dataset = read_training_dataset()
    training_dataset = training_dataset.dropna()
    numeric = split_data(training_dataset)
    target = split_target(training_dataset)
    train_data, test_data, train_target, test_target = train_test_split(numeric, target)
    splited_data = {
        "train_data": train_data,
        "train_target": train_target,
        "test_data": test_data,
        "test_target": test_target,
    }
    return splited_data


def write_mvb_submission():
    pass
