from pathlib import Path

import numpy as np

from .lower_level_utils import get_homeworks_path


def load_dataset(dataset: str):
    data_path: Path = get_homeworks_path() / "data"

    if dataset.lower() != "mnist":
        raise ValueError("HW4 code package only includes the MNIST dataset.")

    with np.load(data_path / "mnist" / "mnist.npz", allow_pickle=True) as f:
        x_train, labels_train = f["x_train"], f["y_train"]
        x_test, labels_test = f["x_test"], f["y_test"]

    x_train = x_train.reshape(-1, 784) / 255
    x_test = x_test.reshape(-1, 784) / 255

    return (x_train, labels_train), (x_test, labels_test)
