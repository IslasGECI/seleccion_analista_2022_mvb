from sklearn.model_selection import train_test_split
from pollos_petrel import read_training_dataset
import pandas as pd
from typing import Tuple


def split_data_target(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = dataset[["target"]]
    numeric = dataset.drop(columns=["target", "id"])
    return numeric, target


def preprocces_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_dataset = read_training_dataset()
    training_dataset = training_dataset.dropna()
    numeric, target = split_data_target(training_dataset)
    split_train_data, split_test_data, train_target, test_target = train_test_split(numeric, target)
    return split_train_data, train_target, split_test_data, test_target


def write_mvb_submission():
    pass
