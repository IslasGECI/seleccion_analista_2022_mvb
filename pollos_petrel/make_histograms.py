import pandas as pd
from matplotlib import pyplot as plt


figure_style = {"figsize": (20, 14)}
plt.figure(figsize=figure_style["figsize"])  # pragma: no mutate


def make_histograms():
    """Crea histogramas de cada columna de nuestros datos train.csv"""
    data = pd.read_csv("pollos_petrel/train.csv")
    data.hist()
    plt.savefig("histogramas.png")
