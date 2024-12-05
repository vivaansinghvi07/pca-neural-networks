from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from net import NormalNetwork, PCANetwork, test_network, train_network

sb.set_theme(style="whitegrid")


# https://stackoverflow.com/a/67992521
def show_image_list(
    list_images,
    list_titles=None,
    list_cmaps=None,
    grid=False,
    num_cols=2,
    figsize=(20, 10),
    title_fontsize=30,
):
    """
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    """

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), "%d imgs != %d titles" % (
            len(list_images),
            len(list_titles),
        )

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), "%d imgs != %d cmaps" % (
            len(list_images),
            len(list_cmaps),
        )

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img = list_images[i]
        title = list_titles[i] if list_titles is not None else None

        list_axes[i].imshow(img, cmap='gray')
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)
        list_axes[i].axis('off')

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()


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
        xlabel="epoch",
        ylabel="metric",
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
        title=f"Time Taken to Train a Network for {epochs} Epochs", ylabel="Time (s)"
    )
