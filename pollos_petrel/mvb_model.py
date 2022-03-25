import pandas as pd
from typing import Tuple

def split_data_target(dataset: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    target = dataset[["target"]]
    numeric = dataset.drop(columns=["target", "id"])
    return numeric, target


def write_mvb_submission():
    pass
