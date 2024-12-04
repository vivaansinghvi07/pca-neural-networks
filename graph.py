from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sb
from net import NormalNetwork, PCANetwork, test_network, train_network


def draw_metric_graph(
    *,
    pca_thresh: list[float],
    epochs: int = 20,
    metric: str = "accuracy",
    unpack_data: bool = True,
) -> None:
    print("Initializing networks...")
    pca_nets = [PCANetwork(pca_thresh=t, unpack_data=unpack_data) for t in pca_thresh]
    normal_net = NormalNetwork(unpack_data=unpack_data)
    pca_results, normal_results = [], []
    for _ in range(epochs):
        for net in pca_nets:
            train_network(net, epochs=1)
        train_network(normal_net, epochs=1)
        pca_results.append(tuple(test_network(net)[metric] for net in pca_nets))
        normal_results.append(test_network(normal_net)[metric])
    plt.plot([*range(epochs)], normal_results, label="Normal Results")
    for res, t in zip(zip(*pca_results), pca_thresh):
        plt.plot([*range(epochs)], res, label=f"PCA (Thresh {t})")
    plt.title(f"{metric.title()} Over {epochs} Epochs with Different Networks")
    plt.legend()
    plt.ylabel('%')
    plt.xlabel('Epoch')
    plt.show()

def draw_time_graph(
    pca_thresh: list[float],
    epochs: int = 20,
    unpack_data: bool = True,
) -> None:
    normal_net = NormalNetwork(unpack_data=unpack_data)
    pca_nets = [PCANetwork(pca_thresh=t, unpack_data=unpack_data) for t in pca_thresh]
    times = []
    for net in [normal_net] + pca_nets:
        start = perf_counter()
        train_network(net, epochs=epochs)
        times.append(perf_counter() - start)
        
    categories = ["normal"] + [f"thresh {t}" for t in pca_thresh]
    sb.set_theme(style="whitegrid")
    sb.barplot( x = categories, y= times, hue = categories)
    plt.title(f"Time Taken to Train a Network for {epochs} Epochs")
    plt.ylabel("Time (s)")
    plt.show()