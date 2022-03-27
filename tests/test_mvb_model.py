from pollos_petrel import split_data, split_target, preprocces_training_data
import pandas as pd


def test_split_data():
    data = {"id": [1, 2], "numeros": [5, 6], "target": [3, 4]}
    dataset = pd.DataFrame(data=data)
    numeric = split_data(dataset)

    obtained_numeric_columns = numeric.shape[1]
    expected_numeric_columns = 1
    assert obtained_numeric_columns == expected_numeric_columns


def test_split_target():
    data = {"id": [1, 2], "numeros": [5, 6], "target": [3, 4]}
    dataset = pd.DataFrame(data=data)
    target = split_target(dataset)
    obtained_target_columns = target.shape[1]
    expected_target_columns = 1
    assert obtained_target_columns == expected_target_columns


def test_preprocces_training_data():
    splited_data = preprocces_training_data()
    keys = ["train_data", "train_target", "test_data", "test_target"]
    for key in splited_data.keys():
        assert key in keys

