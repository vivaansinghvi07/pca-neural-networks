import functools
import typing

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


def cache(func: typing.Callable):
    d = {}

    @functools.wraps(func)
    def inner(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if key not in d:
            d[key] = func(*args, **kwargs)
        return d[key]

    return inner


class PCA:
    """
    Stores information and exposes functions to compute the PCA
    for a given dataset, and is callable as to apply said PCA to
    a given input.
    """

    _hash: int
    _keep_features: np.ndarray
    _base_data_mean: np.ndarray
    _base_data_std_dev: np.ndarray
    _standardized_data: np.ndarray
    _cum_variance_explained: np.ndarray
    _components: np.ndarray

    def __init__(self, data: np.ndarray):
        """Calculates the PCA and stores the results"""

        self._keep_features = data.sum(axis=0) != 0
        base_data = data[:, self._keep_features]
        self._base_data_mean = base_data.mean(axis=0)
        self._base_data_std_dev = base_data.std(axis=0)
        self._standardized_data = (
            base_data - self._base_data_mean
        ) / self._base_data_std_dev

        cov_matrix = np.cov(self._standardized_data.T)
        evalues, evectors = np.linalg.eigh(cov_matrix)
        sort_order = np.flip(evalues.argsort())
        self._components = evectors[:, sort_order]
        self._cum_variance_explained = np.cumsum(evalues[sort_order] / sum(evalues))
        self._hash = id(
            self._components
        )  # arbitrary, just needs to be different per object

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._base_data_mean) / self._base_data_std_dev

    @cache
    def get_n_params(self, thresh: float) -> int:
        return (
            np.argwhere(self._cum_variance_explained >= thresh)[0][0] + 1
        )  # first instance of being over

    @cache
    def _get_projection_components(self, thresh: float = 0.95) -> np.ndarray:
        """Returns the projection vectors for a specific PCA threshold"""
        components = self._components[:, : self.get_n_params(thresh)]
        return components

    @cache
    def _compute_normalization(self, thresh: float) -> typing.Tuple[float, float]:
        """Returns the mean and standard deviation of the feature set at some threshold"""
        projected_data = self._standardized_data.dot(
            self._get_projection_components(thresh)
        )
        return projected_data.mean(), projected_data.std()

    def __call__(self, x: torch.Tensor, thresh: float = 0.95):
        mat = self._get_projection_components(thresh)
        mean, std = self._compute_normalization(thresh)
        standardized_x = self._standardize(x.numpy()[self._keep_features])
        return torch.Tensor((standardized_x.dot(mat) - mean) / std)

    # for caching to work
    def __hash__(self) -> int:
        return self._hash


class PCATransform:
    """
    Small wrapper around PCA supporting the PyTorch transforms API
    """

    pca: PCA
    thresh: float

    def __init__(self, thresh: float) -> None:
        self.thresh = thresh
        self.pca = precompute_mnist_params()["pca"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.pca(x, thresh=self.thresh)


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


def get_mnist_pca(
    train: bool, *, pca_thresh: float = 0.90, batch_size: int = 32
) -> DataLoader:
    data = torchvision.datasets.MNIST(
        "./data",
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torch.flatten, PCATransform(pca_thresh)]
        ),
        train=train,
        download=True,
    )
    return DataLoader(data, batch_size)


def get_mnist_raw(train: bool, batch_size: int = 32) -> DataLoader:
    norm = precompute_mnist_params()["normalization"]
    data = torchvision.datasets.MNIST(
        "./data",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                norm,
                torch.flatten,
            ]
        ),
        train=train,
        download=True,
    )
    return DataLoader(data, batch_size)
