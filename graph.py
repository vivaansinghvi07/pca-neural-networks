from time import perf_counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from net import NormalNetwork, PCANetwork, test_network, train_network

sb.set_theme(style="whitegrid")


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
    df = pd.melt(
        pd.DataFrame(
            {
                "epoch": [*range(epochs)],
                "Normal": normal_results,
            }
            | {
                f"PCA (Thresh {t})": res
                for res, t in zip(zip(*pca_results), pca_thresh)
            }
        ),
        "epoch",
    )
    sb.lineplot(data=df, x="epoch", y="value", hue="variable").set(
        title=f"{metric.title()} Over {epochs} Epochs with Different Networks",
        xlabel="epoch", ylabel="metric"
    )
    plt.legend(title="Method")


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
    sb.barplot(x=categories, y=times, hue=categories).set(
        title=f"Time Taken to Train a Network for {epochs} Epochs",
        ylabel="Time (s)"
    )
