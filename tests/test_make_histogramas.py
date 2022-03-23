from pollos_petrel import data_histogramas
import os


def test_data_histogramas():
    imagen_path = "histogramas.png"
    if os.path.exists(imagen_path):
        os.remove(imagen_path)
    data_histogramas()
    assert os.path.exists(imagen_path)
    os.remove(imagen_path)
