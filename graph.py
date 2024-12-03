import matplotlib.pyplot as plt

from net import NormalNetwork, PCANetwork, test_network, train_network


def draw_metric_graph(
    *,
    pca_thresh: float = 0.90,
    epochs: int = 20,
    metric: str = "accuracy",
    unpack_data: bool = True,
) -> None:
    print("Initializing networks...")
    pca_net = PCANetwork(pca_thresh=pca_thresh, unpack_data=unpack_data)
    normal_net = NormalNetwork(unpack_data=unpack_data)
    pca_results, normal_results = [], []
    for _ in range(epochs):
        train_network(pca_net, epochs=1)
        train_network(normal_net, epochs=1)
        pca_results.append(test_network(pca_net)[metric])
        normal_results.append(test_network(normal_net)[metric])
    plt.plot([*range(epochs)], normal_results, label="Normal Results")
    plt.plot([*range(epochs)], pca_results, label="PCA Results")
    plt.savefig(f"{metric}_over_{epochs}.png")
    plt.legend()
    plt.show()
