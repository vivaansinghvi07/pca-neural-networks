import typing
from functools import wraps

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


def cache(func: typing.Callable):
    d = {}
    @wraps(func)
    def inner(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if key not in d:
            d[key] = func(*args, **kwargs)
        return d[key]
    return inner


class PCA:

    __keep_features: np.ndarray
    __base_data: np.ndarray
    __cum_variance_explained: np.ndarray
    __components: np.ndarray

    def __init__(self, data: np.ndarray):

        self.__keep_features = data.sum(axis=0) != 0
        self.__base_data = data[:, self.__keep_features]
        standardized_features = (
            self.__base_data - self.__base_data.mean(axis=0)
        ) / self.__base_data.std(axis=0)
        cov_matrix = np.cov(standardized_features.T)
        evalues, evectors = np.linalg.eig(cov_matrix)
        self.__components = evectors
        self.__cum_variance_explained = np.cumsum(evalues / sum(evalues))
        self.__hash = id(self.__components)  # arbitrary, just needs to be different per object

    @cache
    def _get_projection_matrix(self, thresh: float = 0.95):
        n_components = (
            np.argwhere(self.__cum_variance_explained >= thresh)[0][0] + 1
        )  # first instance of being over
        components = self.__components[:, :n_components]
        return components @ np.linalg.inv(components.T @ components) @ components.T

    @cache
    def _compute_normalization(
        self, thresh: float
    ) -> typing.Tuple[float, float]:
        projected_data = self._get_projection_matrix(thresh) @ self.__base_data.T
        return projected_data.mean(), projected_data.std()

    def __call__(self, x: torch.Tensor, thresh: float = 0.95):
        mat = self._get_projection_matrix(thresh)
        mean, std = self._compute_normalization(thresh)
        kept_features = x[self.__keep_features]
        return (torch.Tensor(mat) @ kept_features - mean) / std

    # for caching to work
    def __hash__(self) -> int:
        return self.__hash


class PCATransform:
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
