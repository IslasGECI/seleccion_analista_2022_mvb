import pandas as pd
from matplotlib import pyplot as plt

plt.figure(figsize=(20, 14))  # pragma: no mutate


def data_histogramas():
    """Crea histogramas de cada columna de nuestros datos train.csv"""
    data = pd.read_csv("pollos_petrel/train.csv")
    data.hist()
    plt.savefig("histogramas.png")


data_histogramas()
