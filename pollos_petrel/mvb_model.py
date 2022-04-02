from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from pollos_petrel import read_training_dataset, read_testing_dataset
import pandas as pd


def split_data(dataset: pd.DataFrame) -> pd.DataFrame:
    numeric = dataset.drop(columns=["target", "id"])
    return numeric


def split_target(dataset: pd.DataFrame) -> pd.DataFrame:
    target = dataset[["target"]]
    return target


def preprocces_training_data() -> dict:
    training_dataset = read_training_dataset()
    training_dataset = training_dataset.dropna()
    numeric = split_data(training_dataset)
    target = split_target(training_dataset)
    train_data, test_data, train_target, test_target = train_test_split(numeric, target)
    splited_data = {
        "train_data": train_data,
        "train_target": train_target,
        "test_data": test_data,
        "test_target": test_target,
    }
    return splited_data


def preprocces_testing_data(model: Pipeline) -> pd.DataFrame:
    testing_dataset = read_testing_dataset()
    no_nan_dataset = testing_dataset[["id"]].copy()
    imputer = SimpleImputer()
    no_nan_dataset.loc[:, model.feature_names_in_] = imputer.fit_transform(
        testing_dataset.loc[:, model.feature_names_in_]
    )
    return no_nan_dataset


class LinearModel(Pipeline):
    def __init__(self):
        self.splited_data = preprocces_training_data()
        self.model = self.set_regression()
        self.submission_path = "pollos_petrel/mvb_linear_submission.csv"

    def set_regression(self) -> Pipeline:
        """Define y entrena el modelo escogido. Las opciones son:
        *- LinearModel
        *- LogisticModel

        En el modelo linear se usan las columnas 'Longitud_ala' y
                        'Longitu_pluma_exterior_de_la_cola' por ser las variables con
                        una correlación más alta
        """
        model = make_pipeline(StandardScaler(), LinearRegression())
        print(f"Descripción del modelo: {model.steps}")
        model.fit(
            self.splited_data["train_data"][["Longitud_ala", "Longitud_pluma_exterior_de_la_cola"]],
            self.splited_data["train_target"],
        )
        return model

    def write_submission(self):
        get_error_model(self.splited_data, self.model)
        submission = make_predictions(self.model)
        submission.to_csv(self.submission_path)


class LogisticModel(Pipeline):
    def __init__(self):
        self.splited_data = preprocces_training_data()
        self.model = self.set_regression()
        self.submission_path = "pollos_petrel/mvb_logistic_submission.csv"

    def set_regression(self) -> Pipeline:
        model = make_pipeline(StandardScaler(), LogisticRegression())
        print(f"Descripción del modelo: {model.steps}")
        model.fit(
            self.splited_data["train_data"], self.splited_data["train_target"]["target"].values
        )
        return model

    def write_submission(self):
        get_error_model(self.splited_data, self.model)
        submission = make_predictions(self.model)
        submission.to_csv(self.submission_path)


def make_predictions(model: Pipeline) -> pd.DataFrame:
    testing_dataset = preprocces_testing_data(model)
    target_predictions = model.predict(testing_dataset.loc[:, model.feature_names_in_])
    submission = testing_dataset[["id"]].copy()
    submission = submission.assign(target=target_predictions)
    return submission


def get_error_model(splited_data: dict, model: Pipeline) -> float:
    target_predicted = model.predict(splited_data["test_data"][model.feature_names_in_])
    error = mean_absolute_error(target_predicted, splited_data["test_target"])
    print(f"En promedio el error de nuestro modelo es {error:.2f} dias ")
    return error


def write_both_submissions():
    linear = LinearModel()
    linear.write_submission()

    logistic = LogisticModel()
    logistic.write_submission()
