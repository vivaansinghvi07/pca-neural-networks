import tqdm

from net.net1 import test_network, train_network, PCANetwork, NormalNetwork
import torch
import matplotlib.pyplot as plt

def draw_metric_graph(
    normal_train_data: torch.Tensor,
    normal_test_data: torch.Tensor,
    pca_train_data: torch.Tensor,
    pca_test_data: torch.Tensor,
    *,
    epochs: int = 30,
    metric: str = "accuracy"
) -> None:
    pca_net = PCANetwork(pca_train_data.shape[1])
    normal_net = NormalNetwork()
    pca_results, normal_results = [], []
    for _ in tqdm.trange(epochs):
        train_network(pca_net, pca_train_data, epochs=1)
        train_network(normal_net, normal_train_data, epochs=1)
        pca_results.append(test_network(pca_net, pca_test_data))
        normal_results.append(test_network(normal_net, normal_test_data))
    plt.plot([*range(epochs)], pca_results, normal_results)
    plt.savefig(f"{metric}_over_{epochs}.png")

