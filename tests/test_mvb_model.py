from pollos_petrel import split_data_target
import pandas as pd


def test_split_data_target():
    data = {"id": [1, 2], "numeros": [5, 6], "target": [3, 4]}
    dataset = pd.DataFrame(data=data)
    numeric, target = split_data_target(dataset)
    obtained_target_columns = target.shape[1]
    expected_target_columns = 1
    assert obtained_target_columns == expected_target_columns
    obtained_numeric_columns = numeric.shape[1]
    expected_numeric_columns = 1
    assert obtained_numeric_columns == expected_numeric_columns
