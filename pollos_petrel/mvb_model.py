from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from pollos_petrel import read_training_dataset, read_testing_dataset
import pandas as pd


def _split_data(dataset: pd.DataFrame) -> pd.DataFrame:
    numeric = dataset.drop(columns=["target", "id"])
    return numeric


def _split_target(dataset: pd.DataFrame) -> pd.DataFrame:
    target = dataset[["target"]]
    return target


def _preprocess_training_data() -> dict:
    training_dataset = read_training_dataset()
    training_dataset = training_dataset.dropna()
    numeric = _split_data(training_dataset)
    target = _split_target(training_dataset)
    train_data, test_data, train_target, test_target = train_test_split(numeric, target)
    splited_data = {
        "train_data": train_data,
        "train_target": train_target,
        "test_data": test_data,
        "test_target": test_target,
    }
    return splited_data


class General_Model(Pipeline):
    """Define y entrena el modelo escogido. Las opciones son:
    *- LinearModel
    *- LogisticModel

    En el modelo linear se usan las columnas 'Longitud_ala' y
    'Longitud_pluma_exterior_de_la_cola' por ser las variables con una correlación más alta
    """

    def __init__(self):
        self.splited_data = _preprocess_training_data()
        self.model = self.set_regression()
        self._preprocess_testing_data()

    def _preprocess_testing_data(self):
        raw_testing_dataset = read_testing_dataset()
        self.testing_dataset = raw_testing_dataset[["id"]].copy()
        imputer = SimpleImputer()
        self.testing_dataset.loc[:, self.model.feature_names_in_] = imputer.fit_transform(
            raw_testing_dataset.loc[:, self.model.feature_names_in_]
        )

    def make_predictions(self) -> pd.DataFrame:
        target_predictions = self.model.predict(
            self.testing_dataset.loc[:, self.model.feature_names_in_]
        )
        submission = self.testing_dataset[["id"]].copy()
        submission = submission.assign(target=target_predictions)
        return submission

    def get_error_model(self) -> float:
        target_predicted = self.model.predict(
            self.splited_data["test_data"][self.model.feature_names_in_]
        )
        error = mean_absolute_error(target_predicted, self.splited_data["test_target"])
        print(f"En promedio el error de nuestro modelo es {error:.2f} dias ")
        return error

    def write_submission(self):
        self.get_error_model()
        submission = self.make_predictions()
        submission.to_csv(self.submission_path)


class LinearModel(General_Model):
    def __init__(self):
        General_Model.__init__(self)
        self.submission_path = "pollos_petrel/mvb_linear_submission.csv"

    def set_regression(self) -> Pipeline:
        model = make_pipeline(StandardScaler(), LinearRegression())
        print(f"Descripción del modelo: {model.steps}")
        model.fit(
            self.splited_data["train_data"][["Longitud_ala", "Longitud_pluma_exterior_de_la_cola"]],
            self.splited_data["train_target"],
        )
        return model


class LogisticModel(General_Model):
    def __init__(self):
        General_Model.__init__(self)
        self.submission_path = "pollos_petrel/mvb_logistic_submission.csv"

    def set_regression(self) -> Pipeline:
        model = make_pipeline(StandardScaler(), LogisticRegression())
        print(f"Descripción del modelo: {model.steps}")
        model.fit(
            self.splited_data["train_data"], self.splited_data["train_target"]["target"].values
        )
        return model


def write_both_submissions():
    REGRESSION_MODELS_SELECTOR = {"linear": LinearModel(), "logistic": LogisticModel()}
    linear = REGRESSION_MODELS_SELECTOR["linear"]
    linear.write_submission()

    logistic = REGRESSION_MODELS_SELECTOR["logistic"]
    logistic.write_submission()
