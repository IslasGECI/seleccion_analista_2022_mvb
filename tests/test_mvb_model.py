from pollos_petrel import (
    LinearModel,
    LogisticModel,
    write_both_submissions,
)
import os
import pytest


test_data = [
    ("linearregression", LinearModel),  # type: ignore
    ("logisticregression", LogisticModel),  # type: ignore
]


@pytest.mark.parametrize("expected_model,  Model", test_data, ids=["linear", "logistic"])
def test_set_model(expected_model, Model):
    obtained_model = Model().set_regression().steps[1][0]
    assert obtained_model == expected_model


test_data = [(LinearModel), (LogisticModel)]  # type: ignore


@pytest.mark.parametrize("regression", test_data, ids=["with LinearModel", "with LogisticModel"])
def test_make_predictions(regression):
    model = regression()
    predictions = model.make_predictions()
    is_target_null = predictions["target"].isnull().any()
    assert not (is_target_null)
    obtained_colums = predictions.shape[1]
    expected_columns = 2
    assert obtained_colums == expected_columns


@pytest.mark.parametrize("regression", test_data, ids=["for LinearModel", "for  LogisticModel"])
def test_preprocess_testing_data(regression):
    model = regression()
    data = model.testing_dataset
    n_obtained_columns = len(data.columns)
    n_expected_columns = len(model.model.feature_names_in_) + 1
    assert n_obtained_columns == n_expected_columns


def test_get_error_model():
    model = LinearModel()
    obtained_error = model.get_error_model()
    assert obtained_error > 0


def test_write_both_submissions():
    submission_paths = [
        "pollos_petrel/mvb_linear_submission.csv",
        "pollos_petrel/mvb_logistic_submission.csv",
    ]
    for path in submission_paths:
        if os.path.exists(path):
            os.remove(path)

    write_both_submissions()
    for path in submission_paths:
        assert os.path.exists(path) and os.path.exists(path)
