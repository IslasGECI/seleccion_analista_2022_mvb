from pollos_petrel import make_histograms, figure_style
import os


def test_figure_style():
    expected_style = {"figsize": (20, 14)}
    obtained_style = figure_style
    assert obtained_style == expected_style


def test_make_histograms():
    imagen_path = "histogramas.png"
    if os.path.exists(imagen_path):
        os.remove(imagen_path)
    make_histograms(figure_style)
    assert os.path.exists(imagen_path)
    os.remove(imagen_path)
