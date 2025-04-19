from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def bubble_plots(ax: plt.Axes, feature_name: str, vmin_: float, vmax_: float):
    """Bubble plot the meadian of the feature_name in each cluster."""
    df = pd.read_csv(f"./data/state_clustered_dataframe.csv")
    df["mRS_diff"] = df["mRS_after"] - df["mRS_before"]
    scatter_matrix = np.load(f"./data/state_scatter_matrix.npz")["arr_0"]

    cluster_size = df.groupby("state_class").size()
    cluster_centers = []
    for state in cluster_size.index:
        state_indice = df[df.state_class == state].index.tolist()
        cluster_centers.append(scatter_matrix[state_indice].mean(axis=0))
    cluster_centers = np.array(cluster_centers)
    medians = df.groupby("state_class")[feature_name].median()
    if feature_name == "mRS_before":
        medians -= 1
    unique_values = np.sort(df[feature_name].unique())
    num_values = len(unique_values)
    min_value = min(unique_values)
    print(medians.describe())

    # width = 9 if feature_name == "age" else 8
    if feature_name == "age":
        color_ = "jet"
    elif feature_name == "SAH_severity":
        color_ = "Reds"
    elif feature_name == "max_diameter":
        color_ = "Greens"
    elif feature_name == "aneurysm_site":
        color_ = "Paired"
    elif feature_name == "mRS_diff":
        color_ = "jet"
    else:
        color_ = "Blues"
    # plt.style.use("ggplot")
    scatter = ax.scatter(
        x=cluster_centers[:, 0],
        y=cluster_centers[:, 1],
        s=cluster_size,
        alpha=0.9,
        sizes=(50, 500),
        c=medians,
        cmap=color_,
        linewidths=0.2,
        edgecolors="gray",
        vmin=vmin_,
        vmax=vmax_,
    )

    if feature_name != "age":
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="lower right", title=feature_name
        )
        ax.add_artist(legend1)

    ax.set_xlabel("UMAPr1")
    ax.set_ylabel("UMAPr2")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f"{feature_name.replace('_',' ')}", fontsize=30)

    return scatter


def cluster_feature_heatmap():
    df = pd.read_csv("./data/state_clustered_dataframe.csv")
    df.gender -= 1
    target_cols = [
        "age",
        "gender",
        "mRS_before",
        "SAH_severity",
        "max_diameter",
    ]
    print(df[target_cols].describe())
    df[target_cols] = MinMaxScaler().fit_transform(df[target_cols])
    counts = []
    groups = df.groupby("state_class")
    for col in target_cols:
        counts.append(groups[col].mean())
    counts = np.array(counts)
    # counts = np.sort(counts, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(counts, cmap="Blues", ax=ax, robust=True)
    # sns.histplot(data=df, x="age", ax=ax)
    ax.set_yticklabels(target_cols)
    ax.set_xlabel("Cluster ID")
    fig.savefig("./fig/clusters/cluster_feature_heatmap.pdf")


def scatter_cutoff(group_name, cutoffs: list[tuple[str, Union[float, tuple], bool]]):
    """
    Scatter plot the samples that are cut off by the cutoffs.

    Args:
        group_name: str
        cutoffs: list of tuples of (feature_name, cutoff, leq).
            If cutoff is tuple, the values are used as isin.
    """

    def limit_range(df: pd.DataFrame, feature_name, cutoff: float, leq: bool):
        index = df[feature_name] <= cutoff if leq else df[feature_name] > cutoff
        return df.loc[index]

    df = pd.read_csv(f"./data/state_clustered_dataframe.csv")
    df["mRS_before"] -= 1
    df_1 = df.copy()
    scatter_matrix = np.load(f"./data/state_scatter_matrix.npz")["arr_0"]

    for feature_name, cutoff_, leq in cutoffs:
        if isinstance(cutoff_, tuple):
            if leq == True:
                df_1 = df_1[df_1[feature_name].isin(cutoff_)]
            else:
                df_1 = df_1[~df_1[feature_name].isin(cutoff_)]
        else:
            df_1 = limit_range(df_1, feature_name, cutoff_, leq)

    print(group_name, len(df_1))

    def scatter_samples(matrix, ax, alpha_: float, color: str, label_: str):
        ax.scatter(
            matrix[:, 0],
            matrix[:, 1],
            alpha=alpha_,
            s=10,
            c=color,
            label=label_,
        )

    fig, ax = plt.subplots(figsize=(5, 5))
    scatter_samples(scatter_matrix, ax, 0.1, "gray", "all")
    scatter_samples(
        scatter_matrix[df_1.index],
        ax,
        1,
        "red",
        "cutoff",
    )
    ax.set_title(group_name)

    plt.show()
    fig.savefig(f"./fig/clusters/{group_name}.pdf")


def plot_feature_clusters():
    fig, axes = plt.subplots(1, 5, figsize=(25, 10))
    features = [
        ["age", 20, 90],
        ["SAH_severity", 0, 6],
        ["mRS_before", 0, 5],
        ["max_diameter", 0, 5],
        ["aneurysm_site", 0, 18],
    ]
    for i, (feature_name, vmin, vmax) in enumerate(features):
        scatter = bubble_plots(axes[0, i], feature_name, vmin, vmax)
        if feature_name == "age":
            fig.colorbar(scatter, ax=axes[0, i], pad=0.02)
    plt.show()
    fig.savefig(f"./fig/clusters/bubble_plots.pdf")


if __name__ == "__main__":
    plot_feature_clusters()

    cilostazol_11 = [
        ("age", 73, False),
        ("age", 78, True),
        ("mRS_before", 1, True),
        ("SAH_severity", (3,), True),
    ]
    cilostazol_18 = [
        ("age", 68, True),
        ("mRS_before", 3, False),
        ("SAH_severity", 3, False),
    ]

    for group_name, cutoffs in zip(
        ("cilostazol_11", "cilostazol_18"), (cilostazol_11, cilostazol_18)
    ):
        scatter_cutoff(group_name, cutoffs)
