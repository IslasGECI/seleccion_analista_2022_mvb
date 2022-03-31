from pollos_petrel import (
    split_data,
    split_target,
    preprocces_training_data,
    set_model,
    LinearModel,
    LogisticModel,
    make_predictions,
    get_error_model,
    write_both_submissions,
)
import os
import pandas as pd
import pytest


@pytest.fixture
def setup_test_split(split_something):
    data = {"id": [1, 2], "numeros": [5, 6], "target": [3, 4]}
    dataset = pd.DataFrame(data=data)
    splited = split_something(dataset)
    return splited


test_data = [(split_data), (split_target)]


@pytest.mark.parametrize("split_something", test_data, ids=["spliting_data", "spliting_target"])
def test_split_something(setup_test_split):
    expected_columns = 1
    obtained_colums = setup_test_split.shape[1]
    assert expected_columns == obtained_colums


def test_preprocces_training_data():
    splited_data = preprocces_training_data()
    keys = ["train_data", "train_target", "test_data", "test_target"]
    for key in splited_data.keys():
        assert key in keys


testdata = [
    ("linearregression", LinearModel),
    ("logisticregression", LogisticModel),
]


@pytest.mark.parametrize("expected_model,  Model", testdata, ids=["linear", "logistic"])
def test_set_linear_regression(expected_model, Model):
    splited_data = preprocces_training_data()
    obtained_model = set_model(splited_data, Model).steps[1][0]
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


def test_get_error_model():
    splited_data = preprocces_training_data()
    model = set_model(splited_data, LinearModel)
    obtained_error = get_error_model(splited_data, model)
    assert obtained_error > 0


test_data = [
    (LinearModel, "pollos_petrel/mvb_linear_submission.csv"),  # type: ignore
    (LogisticModel, "pollos_petrel/mvb_logistic_submission.csv"),  # type: ignore
]


@pytest.fixture
def setup_test_write_submission(submission_path, regression):
    if os.path.exists(submission_path):
        os.remove(submission_path)
    model = regression()
    model.write_submission()
    submission = pd.read_csv(submission_path)
    return submission


@pytest.mark.parametrize(
    "regression,submission_path",
    test_data,
    ids=["Escribiendo LinearModel", "Escribiendo LogisticModel"],
)
def test_write_submission(setup_test_write_submission):
    submission = setup_test_write_submission
    submission_rows = submission.shape[0]
    assert submission_rows > 1


test_data = [
    ("pollos_petrel/mvb_linear_submission.csv"),  # type: ignore
    ("pollos_petrel/mvb_logistic_submission.csv"),  # type: ignore
]


@pytest.mark.parametrize(
    "submission_path",
    test_data,
    ids=["Probando LinearModel", "Probando LogisticModel"],
)
def test_write_both_submissions(submission_path):
    if os.path.exists(submission_path):
        os.remove(submission_path)
    write_both_submissions()
    assert os.path.exists(submission_path)
