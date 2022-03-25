import pandas as pd


def split_data_target(dataset: pd.DataFrame) -> pd.DataFrame:
    target = dataset[["target"]]
    numeric = dataset.drop(columns=["target", "id"])
    return numeric, target


def write_mvb_submission():
    pass
