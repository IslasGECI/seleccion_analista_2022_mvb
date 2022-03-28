from pollos_petrel import (
    split_data,
    split_target,
    preprocces_training_data,
    set_model,
    LinearModel,
    LogisticModel,
    make_predictions,
)
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


def test_set_linear_regression():
    splited_data = preprocces_training_data()
    expected_model = "linearregression"
    obtained_model = set_model(splited_data, LinearModel).steps[1][0]
    assert obtained_model == expected_model


def test_set_logistic_regression():
    splited_data = preprocces_training_data()
    expected_model = "logisticregression"
    obtained_model = set_model(splited_data, LogisticModel).steps[1][0]
    assert obtained_model == expected_model


def test_make_predictions():
    splited_data = preprocces_training_data()
    model = set_model(splited_data, LinearModel)
    predictions = make_predictions(model)
    is_target_null = predictions["target"].isnull().any()
    assert not (is_target_null)
    obtained_colums = predictions.shape[1]
    expected_columns = 2
    assert obtained_colums == expected_columns
