import typing
from functools import wraps

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


class PCA:

    keep_features: np.ndarray
    _cum_variance_explained: np.ndarray
    _components: np.ndarray

    def __init__(self, data: np.ndarray):
        self.keep_features = data.sum(axis=0) != 0
        nonzero_features = data[:, self.keep_features]
        standardized_features = (
            nonzero_features - nonzero_features.mean(axis=0)
        ) / nonzero_features.std(axis=0)
        cov_matrix = np.cov(standardized_features.T)
        evalues, evectors = np.linalg.eig(cov_matrix)
        self._components = evectors
        self.cum_variance_explained = np.cumsum(evalues / sum(evalues))

    def get_components(self, thresh: float = 0.95):
        n_components = np.argwhere(
            self.cum_variance_explained >= thresh
        ) [0][0] + 1 # first instance of being over
        return torch.Tensor(self._components[:, :n_components])

class PCATransform:
    pca_info: PCA 
    thresh: float

    def __init__(self, thresh: float) -> None:
        self.thresh = thresh

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


def cache(func: typing.Callable):
    d = {}

    @wraps(func)
    def inner(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if key not in d:
            d[key] = func(*args, **kwargs)
        return d[key]

    return inner


@cache
def precompute_mnist_params():
    data = torchvision.datasets.MNIST(
        "./data",
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    complete_dataset: torch.Tensor = next(iter(DataLoader(data, len(data))))[0]

    return {
        "normalization": torchvision.transforms.Normalize(
            [complete_dataset.mean(dim=[0, 2, 3]).item()],
            [complete_dataset.std(dim=[0, 2, 3]).item()],
        ),
        "pca": PCA(complete_dataset.flatten(start_dim=1).numpy()),
    }


def get_mnist_pca(train: bool, *, pca_thresh: float = 0.90, batch_size: int = 32) -> DataLoader:
    data = torchvision.datasets.MNIST(
        "./data",
        transform= torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torch.flatten,
            PCATransform(pca_thresh)
        ]),
        train=train,
        download=True,
    )
    return DataLoader(data, batch_size)


def get_mnist_raw(train: bool, batch_size: int = 32) -> DataLoader:
    norm = precompute_mnist_params()['normalization']
    data = torchvision.datasets.MNIST(
        "./data",
        transform= torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            norm,
            torch.flatten,
        ]),
        train=train,
        download=True,
    )
    return DataLoader(data, batch_size)
