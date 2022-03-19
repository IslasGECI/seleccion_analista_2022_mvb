import dummy_model as dm

trainDF = dm.read_training_dataset()

trainDF.describe().to_csv("describe_train.csv")
