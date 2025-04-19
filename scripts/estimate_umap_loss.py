import datetime
import math

import joblib
import numpy as np
import pandas as pd
import umap

dist_matrix = np.load("./data/state_dist_matrix.npz")["arr_0"]


def estimate_umap_loss(fitted_reducer: umap.UMAP, reduced_matrix: np.ndarray):
    a = fitted_reducer._a
    b = fitted_reducer._b
    graph = fitted_reducer.graph_
    epsilon = 1e-10

    losses = []
    for i, row_matrix in enumerate(graph):
        for col in row_matrix.indices:
            # calc only upper triangular matrix
            if col >= i:
                mu = graph[i, col]
                nu = 1 / (
                    1
                    + a
                    * np.linalg.norm(reduced_matrix[i] - reduced_matrix[col]) ** (2 * b)
                )
                loss = mu * math.log(mu / nu + epsilon) + (1 - mu) * math.log(
                    (1 - mu) / (1 - nu) + epsilon
                )
                losses.append(loss)
    return np.array(losses).mean()


def main(k: int):
    n_neighbors = 50

    reducer = umap.UMAP(
        metric="precomputed",
        min_dist=0,
        n_neighbors=n_neighbors,
        n_components=k,
    )
    matrix = reducer.fit_transform(dist_matrix)
    loss = estimate_umap_loss(reducer, matrix)
    print(k, loss)
    return (k, loss)


if __name__ == "__main__":
    trial_list = list(range(2, 10)) * 10
    results = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(main)(k) for k in trial_list
    )
    pd.DataFrame(results, columns=["n_components", "loss"]).to_csv(
        f"./data/umap_loss_{datetime.datetime.now().isoformat()}.csv", index=False
    )
