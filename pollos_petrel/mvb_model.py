from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pollos_petrel import read_training_dataset, read_testing_dataset, drop_all_but_id
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


class LinearModel(Pipeline):
    def __init__(self, splited_data):
        self.splited_data = splited_data

    def set_regression(self) -> Pipeline:
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(
            self.splited_data["train_data"][["Longitud_ala", "Longitud_pluma_exterior_de_la_cola"]],
            self.splited_data["train_target"],
        )
        return model


class LogisticModel(Pipeline):
    def __init__(self, splited_data):
        self.splited_data = splited_data

    def set_regression(self) -> Pipeline:
        model = make_pipeline(StandardScaler(), LogisticRegression())
        model.fit(
            self.splited_data["train_data"], self.splited_data["train_target"]["target"].values
        )
        return model


def set_model(splited_data: dict, RegressionModel) -> Pipeline:
    """Define y entrena el modelo escogido. Las opciones son:
    *- LinearModel
    *- LogisticModel

    En el modelo linear se usan las columnas 'Longitud_ala' y
            'Longitu_pluma_exterior_de_la_cola' por ser las variables con
            una correlación más alta
    """
    model = RegressionModel(splited_data).set_regression()

    print(f"Modelo seleccionado: {model.steps}")
    return model


def make_predictions(model: Pipeline) -> pd.DataFrame:
    testing_dataset = read_testing_dataset()
    testing_dataset = testing_dataset.dropna()
    submission = drop_all_but_id(testing_dataset)
    target_predictions = model.predict(testing_dataset[model.feature_names_in_])
    submission = submission.assign(target=target_predictions)
    return submission


def get_error_model(splited_data: dict, model: Pipeline) -> float:
    target_predicted = model.predict(splited_data["test_data"][model.feature_names_in_])
    error = mean_absolute_error(target_predicted, splited_data["test_target"])
    print(f"En promedio el error de nuestro modelo es {error:.2f} dias ")
    return error


def write_mvb_submission(RegressionModel):
    """Define el modelo que quieres usar:
    *. LogisticModel
    *. LinearModel

    LinearModel es el mejor
    """
    splited_data = preprocces_training_data()
    model = set_model(splited_data, RegressionModel)
    get_error_model(splited_data, model)
    submission_path = "pollos_petrel/mvb_submission.csv"
    submission = make_predictions(model)
    submission.to_csv(submission_path)


def write_linear_submission():
    write_mvb_submission(LinearModel)


def write_logistic_submission():
    write_mvb_submission(LogisticModel)
