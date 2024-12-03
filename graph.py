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
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def draw_dataset_size_graph(
    pca_thresh: float = 0.90,
    metric: str = "accuracy",
    epochs: int = 2,
):
    pca_results, normal_results = [], []
    sizes = [100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000]
    for size in sizes:
        pca_net = PCANetwork(pca_thresh=pca_thresh)
        normal_net = NormalNetwork()
        train_network(pca_net, epochs=epochs, data_cutoff=size)
        train_network(normal_net, epochs=epochs, data_cutoff=size)
        pca_results.append(test_network(pca_net)[metric])
        normal_results.append(test_network(normal_net)[metric])
    plt.plot(sizes, normal_results, label="Normal Results")
    plt.plot(sizes, pca_results, label="PCA Results")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
