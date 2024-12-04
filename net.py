import typing
from abc import ABCMeta
from math import log2

import torch
import tqdm
from sklearn.metrics import confusion_matrix
from torch import nn

from data import get_mnist_pca, get_mnist_raw, precompute_mnist_params


class Network(nn.Module, metaclass=ABCMeta):
    batch_size: int
    forward_prob: nn.Module
    train_loader: typing.Iterable[typing.Tuple[torch.Tensor, torch.Tensor]]
    test_loader: typing.Iterable[typing.Tuple[torch.Tensor, torch.Tensor]]
    train_data_len: int
    test_data_len: int
    loss_func: torch.nn.CrossEntropyLoss


class NormalNetwork(Network):
    """
    Input shape: (N, 728)  -- the flattened images
    Output shape: (N, 10)  -- the class probabilities
    """

    def __init__(self, batch_size: int = 32, unpack_data: bool = False):
        # normalNetWork inherits nn.Module's attributes
        super(NormalNetwork, self).__init__()
        self.batch_size = batch_size
        self.forward_prob = nn.Sequential(
            # input is a batch of 28x28 images, by default - weights are initialized randomly between [ -sqrt(1/784), sqrt(1/784) ] and biases are initialized to zero
            nn.Linear(pow(28, 2), 512),
            nn.ReLU(),  # hidden layer 1, relu => max(0, x)
            nn.Linear(512, 512),
            nn.ReLU(),  # hidden layer 2
            nn.Linear(512, 10),
        )
        self.train_loader = get_mnist_raw(train=True, batch_size=batch_size)
        self.test_loader = get_mnist_raw(train=False, batch_size=batch_size)
        self.train_data_len = len(self.train_loader.dataset)  # type: ignore
        self.test_data_len = len(self.test_loader.dataset)  # type: ignore
        if unpack_data:
            self.train_loader = [*self.train_loader]
            self.test_loader = [*self.test_loader]

        # applies LogSoftmax then NLLLoss
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_x):
        return self.forward_prob(input_x)


class PCANetwork(Network):
    """
    Input shape: (N, x)    -- the principal components
    Output shape: (N, 10)  -- class probabilities
    """

    def __init__(
        self, pca_thresh: float = 0.9, batch_size: int = 32, unpack_data: bool = False
    ):
        super(PCANetwork, self).__init__()
        self.batch_size = batch_size
        n_params = precompute_mnist_params()["pca"].get_n_params(pca_thresh)
        hidden_layer_dim = 2 ** int(log2(n_params) - 1)
        self.forward_prob = nn.Sequential(
            nn.Linear(n_params, hidden_layer_dim),
            nn.ReLU(),  # hidden layer 1, relu => max(0, x)
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),  # hidden layer 2
            nn.Linear(hidden_layer_dim, 10),
        )
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = get_mnist_pca(
            train=True, pca_thresh=pca_thresh, batch_size=batch_size
        )
        self.test_loader = get_mnist_pca(
            train=False, pca_thresh=pca_thresh, batch_size=batch_size
        )
        self.train_data_len = len(self.train_loader.dataset)  # type: ignore
        self.test_data_len = len(self.test_loader.dataset)  # type: ignore
        if unpack_data:
            self.train_loader = [*self.train_loader]
            self.test_loader = [*self.test_loader]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_prob(x)


def train_network(
    net: Network,
    *,
    save_weights: typing.Optional[str] = None,
    epochs: int = 20,
    learning_rate: float = 0.01,
    data_cutoff: typing.Optional[int] = None,
) -> None:
    """
    Train the neural network using `train_data`.
    `train_data` will be in the shape that the neural network requires.
    For example, if training NormalNetwork, `train_data` will be in the
    shape (N, 728). Batching is up to you, and the train_data is separated into iterable batches.
    """
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(
            tqdm.tqdm(
                net.train_loader,
                f"Training epoch {epoch + 1}",
                total=min(len(net.train_loader), ((data_cutoff or 100000000) + net.batch_size) // net.batch_size),  # type: ignore
            )
        ):
            if data_cutoff is not None and (net.batch_size * i) > data_cutoff:
                break

            # initialize gradients to zero
            optimizer.zero_grad()
            # Forward pass - inexplicit call to NormalNetwork.forward()
            outputLayer = net(images)
            outputLayer = outputLayer.view(-1, 10)

            loss = net.loss_func(outputLayer, labels)
            # backward propagation
            loss.backward()
            # adjust weights and biases
            optimizer.step()

    # save weights and biases for testing
    if save_weights is not None:
        torch.save(net.state_dict(), f"{save_weights}.pth")


def test_network(
    net: NormalNetwork | PCANetwork, *, load_weights: typing.Optional[str] = None
) -> dict[str, typing.Any]:
    """
    Test the neural network using `test_data`
    `test_data` follows the same shape as `train_data` as above.
    Return a dictionary of strings with measurements of network
    performance, like accuracy, f1 score, precision, etc.
    """
    # load weights and biases from training
    if load_weights is not None:
        net.load_state_dict(torch.load(f"{load_weights}.pth"))

    # initialize variables
    total_correct = 0
    total_loss = 0
    model_predicted = []
    actual_labels = []

    # iterate through test data without backpropagation and gradient descent
    with torch.no_grad():
        for images, labels in tqdm.tqdm(net.test_loader, "Testing network"):
            # Forward pass
            outputLayer = net(images)

            # sum loss over each batch
            total_loss += net.loss_func(outputLayer, labels).item()

            # if index with max prob is the same index where actual label is 1, then increment total_correct
            predictedLabels = torch.argmax(outputLayer, dim=1)
            labelsMatrix = torch.zeros(labels.shape[0], 10)
            for im_idx in range(labels.shape[0]):
                labelsMatrix[im_idx][labels[im_idx]] = 1
            actualLabels = torch.argmax(labelsMatrix, dim=1)

            total_correct += (predictedLabels == actualLabels).sum().item()
            model_predicted.extend(predictedLabels.tolist())
            actual_labels.extend(actualLabels.tolist())
            # print(f"Correct predictions: {(predictedLabels == actualLabels).sum().item()}/{len(images)}")

    return {
        "accuracy": total_correct
        / net.test_data_len,  # number of correct predictions / num samples in test dataset
        "loss": total_loss / len(net.test_loader),  # type: ignore | summed loss / number of batches
        "confusion matrix": confusion_matrix(actual_labels, model_predicted),
    }
