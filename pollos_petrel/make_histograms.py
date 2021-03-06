import pandas as pd
from matplotlib import pyplot as plt


figure_style = {"figsize": (20, 14)}


def make_histograms(figure_style):
    """Crea histogramas de cada columna de nuestros datos train.csv"""
    data_filename = "pollos_petrel/train.csv"
    data = pd.read_csv(data_filename)
    data.hist(**figure_style)
    histogram_filename = "histogramas.png"
    plt.savefig(histogram_filename)
    plt.close()
