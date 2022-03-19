import pollos_petrel.dummy_model as dm

trainDF = dm.read_training_dataset()

trainDF.describe().to_csv("pollos_petrel/describe_train.csv")
