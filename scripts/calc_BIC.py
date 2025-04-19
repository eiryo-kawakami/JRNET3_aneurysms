import datetime
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture

BIC_DIR = Path("./data/BIC")
BIC_DIR.mkdir(parents=True, exist_ok=True)


def get_bic(X: np.ndarray, k: int):
    seed = random.randint(0, 1000)
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        init_params="k-means++",
        random_state=seed,
    )
    gmm.fit(X)
    return k, gmm.bic(X)


def main(dist_matrix: np.ndarray, reducer: umap.UMAP):
    clustering_matrix = reducer.fit_transform(dist_matrix)
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_bic)(clustering_matrix, k) for k in range(70, 100)
    )
    pd.DataFrame(results, columns=["k", "BIC"]).to_csv(
        BIC_DIR / f"BIC_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        index=False,
    )


if __name__ == "__main__":
    dist_matrix = np.load("./data/state_dist_matrix.npz")["arr_0"]
    n_neighbors = 50
    n_components = 7

    reducer = umap.UMAP(
        metric="precomputed",
        densmap=False,
        min_dist=0,
        n_neighbors=n_neighbors,
        n_components=n_components,
    )

    num_iterations = 50
    for i in range(num_iterations):
        print(f"Iteration {i}")
        main(dist_matrix, reducer)
