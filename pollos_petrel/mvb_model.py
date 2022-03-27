from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from pollos_petrel import read_training_dataset
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


def set_linear_regression(splited_data: dict) -> Pipeline:
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(
        splited_data["train_data"][["Longitud_ala", "Longitud_pluma_exterior_de_la_cola"]],
        splited_data["train_target"],
    )
    return model


def set_logistic_regression(splited_data: dict) -> Pipeline:
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(splited_data["train_data"], splited_data["train_target"]["target"].values)
    return model


def set_model(splited_data: dict, regression_type: int) -> Pipeline:
    """Define y entrena el modelo escogido. Las opciones son:
    0: LogisticRegression
    1: LinearRegression

    En el modelo linear se usan las columnas 'Longitud_ala' y
            'Longitu_pluma_exterior_de_la_cola' por ser las variables con
            una correlación más alta
    """
    if regression_type == "linearregression":
        model = set_linear_regression(splited_data)
    elif regression_type == "logisticregression":
        model = set_logistic_regression(splited_data)

    print(f"Modelo seleccionado: {model.steps}")
    return model


def write_mvb_submission():
    pass
