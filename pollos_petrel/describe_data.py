from pollos_petrel import read_training_dataset


def describe_data():
    read_training_dataset().describe().to_csv("describe_train.csv")
