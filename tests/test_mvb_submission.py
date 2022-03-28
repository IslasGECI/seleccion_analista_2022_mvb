from pollos_petrel import write_mvb_submission, LinearModel
import os
import pandas as pd


def test_write_mvb_submission():
    submission_path = "pollos_petrel/mvb_submission.csv"
    if os.path.exists(submission_path):
        os.remove(submission_path)
    write_mvb_submission(LinearModel)
    submission = pd.read_csv(submission_path)
    submission_rows = submission.shape[0]
    assert submission_rows > 1
    assert os.path.exists(submission_path)
    os.remove(submission_path)
