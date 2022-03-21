from pollos_petrel import describe_data
import os


def test_describe_data():
    description_path = "describe_train.csv"
    if os.path.exists(description_path):
        os.remove(description_path)
    describe_data()
    assert os.path.exists(description_path)
    os.remove(description_path)
