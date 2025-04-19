from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ProxMatrix
import umap
import utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

seed = 123
DATA_DIR = Path("./data")
FIG_DIR = Path("./fig")


def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the initial data files"""
    common_df = pd.read_csv(DATA_DIR / "common_imputed.csv")
    renamer = utils.get_en_renamer()
    original_common_df = pd.read_excel(
        DATA_DIR
        / "JRNET3_DATA_20210929_脳神経血管内治療学会より/00_治療情報_解析用.xlsx"
    )
    original_common_df = original_common_df.rename(renamer, axis=1)

    embolization_state = pd.read_csv(DATA_DIR / "embolization_state.csv")
    embolization_state.treatment_ID = embolization_state.treatment_ID.str[-1].astype(
        np.uint8
    )
    embolization_state = embolization_state.drop(
        ["data_ID", "angiography_device", "3D_angiography_device"], axis=1
    )

    common_df = common_df.drop(
        [
            "data_ID",
            "complication",
            "complication_outcome",
            "facility",
            "adverse_event",
            "adverse_event_detail",
            "adverse_event_relevance",
        ],
        axis=1,
    )
    common_df = common_df.loc[:, :"anesthesia_5"]
    common_df["treatment_ID"] = original_common_df.treatment_ID

    return common_df, embolization_state


def prepare_dataset(
    common_df: pd.DataFrame, embolization_state: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Filter and prepare the dataset for analysis"""
    # Extract cases treated with only embolization
    X = pd.merge(
        embolization_state,
        common_df[(common_df.treatment == 1) & (common_df.num_other_treatments == 1)],
        how="inner",
        on=["info_ID", "treatment_ID"],
    )
    # Exclude cases with unregistered or unknown mRS_before, mRS_after
    X = X[
        (X.mRS_before != 7)
        & (X.mRS_before != 0)
        & (X.mRS_after != 8)
        & (X.mRS_after != 0)
    ]
    # Exclude cases with multiple aneurysm locations
    X = X.loc[~X.aneurysm_site.str.contains(",")]
    X.aneurysm_site = X.aneurysm_site.astype(int)

    other_info_columns = ["info_ID", "treatment_ID", "treat_date", "technical_success"]
    X_other_info = X[other_info_columns]

    X = X.drop(
        other_info_columns
        + [
            "treatment",
            "num_other_treatments",
            "disease_0",
            "disease_4",
            "disease_10",
            "disease_11",
            "disease_14",
        ]
        + list(X.filter(like="anesthesia_")),
        axis=1,
    )

    y = X.mRS_after - X.mRS_before
    X = X.drop("mRS_after", axis=1)

    rupture_index = X[X.disease_1 == 1].index
    rupture_X = X.loc[rupture_index].drop(list(X.filter(like="disease_")), axis=1)
    rupture_y = y.loc[rupture_index]

    X = rupture_X
    y = rupture_y
    X_other_info = X_other_info.loc[rupture_index]

    return X, y, X_other_info


def get_RF_leaves(
    X: pd.DataFrame, y: pd.Series, depth: int = 15, n_tree: int = 1000
) -> tuple[RandomForestRegressor, np.ndarray]:
    """Create Random Forest model and get leaf assignments"""
    rf = RandomForestRegressor(
        n_estimators=n_tree, max_depth=depth, n_jobs=-1, oob_score=True
    )
    rf.fit(X, y)
    return rf, rf.apply(X)


def estimate_random_forest_dissimilarity(
    X: pd.DataFrame, y: pd.Series, holdout_eval: bool = False, n_tree: int = 1000
) -> tuple[RandomForestRegressor, np.ndarray, pd.Series]:
    """Build Random Forest model, evaluate and get proximity matrix"""
    if holdout_eval:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        model, leaves = get_RF_leaves(X_train, y_train, n_tree=n_tree)
        y_pred = model.predict(X_test)
        print(root_mean_squared_error(y_test, y_pred))
    else:
        model, leaves = get_RF_leaves(X, y, n_tree=n_tree)
        print(model.oob_score_)

    # Generate and save proximity/distance matrix
    u_prox_matrix = ProxMatrix.get_upper_prox_matrix(leaves.astype(np.uint32))
    dist_matrix = ProxMatrix.get_dist_matrix(u_prox_matrix, n_tree)
    np.savez_compressed(DATA_DIR / "state_dist_matrix.npz", dist_matrix)

    # Plot and save feature importances
    feature_importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, 10))
    feature_importances.plot(kind="barh", ax=ax)
    fig.savefig(FIG_DIR / "feature_importances.pdf", bbox_inches="tight")

    return model, dist_matrix, feature_importances


def perform_dimensionality_reduction(
    dist_matrix: np.ndarray, n_neighbors: int
) -> tuple[np.ndarray, int]:
    """Perform UMAP dimensionality reduction

    Notes:
        Run `estimate_umap_loss.py` before running this function
    """
    # Generate 2D matrix for visualization
    dens_reducer = umap.UMAP(
        metric="precomputed",
        densmap=True,
        min_dist=0,
        dens_lambda=1,
        n_neighbors=n_neighbors,
    )
    scatter_matrix = dens_reducer.fit_transform(dist_matrix)
    if not isinstance(scatter_matrix, np.ndarray):
        raise ValueError("scatter_matrix is not a numpy array")
    np.savez_compressed(DATA_DIR / "state_scatter_matrix.npz", scatter_matrix)

    # Determine optimal dimensions for UMAP
    loss_results = pd.read_csv(DATA_DIR / "umap_loss.csv")
    loss_results.columns = ["n_components", "loss"]
    loss_means = loss_results.groupby("n_components").mean()
    optimal_umap_k = int(utils.L_method(loss_means))

    # Reduce dimensions to optimal_umap_k
    reducer = umap.UMAP(
        metric="precomputed",
        densmap=False,
        min_dist=0,
        n_neighbors=n_neighbors,
        n_components=optimal_umap_k,
    )
    dens_reduced_matrix = reducer.fit_transform(dist_matrix)
    if not isinstance(dens_reduced_matrix, np.ndarray):
        raise ValueError("dens_reduced_matrix is not a numpy array")

    return dens_reduced_matrix, optimal_umap_k


def estimate_optimal_cluster_numbers(bic_dir: Path) -> int:
    """Estimate the optimal number of clusters"""
    k_bic = []
    for path in bic_dir.glob("*.csv"):
        array = pd.read_csv(path).values
        k_bic.extend(array)
    df_bic = pd.DataFrame(k_bic, columns=["k", "bic"])

    bic_mean = df_bic.groupby("k").mean()
    optimal_k = int(bic_mean.index[np.argmin(bic_mean)])  # type:ignore
    print(f"Optimal clustering number: {optimal_k}")

    return optimal_k


def perform_clustering(
    reduced_matrix: np.ndarray,
    X: pd.DataFrame,
    X_other_info: pd.DataFrame,
    y: pd.Series,
    optimal_k: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Perform clustering and save results"""
    clusters = GaussianMixture(
        n_components=optimal_k, init_params="k-means++", random_state=seed
    ).fit_predict(reduced_matrix)

    assert (
        reduced_matrix.shape[0] == X_other_info.shape[0]
    ), "clustering matrix and X_other_info must have the same number of rows"

    X_clustered = X.copy(deep=True)
    X_clustered["mRS_after"] = y + X.mRS_before
    X_clustered[X_other_info.columns] = X_other_info
    X_clustered["state_class"] = clusters
    X_clustered.to_csv(DATA_DIR / "state_clustered_dataframe.csv", index=False)

    return X_clustered, clusters


def main() -> None:
    n_tree = 1000
    n_neighbors = 50

    common_df, embolization_state = load_and_preprocess_data()

    X, y, X_other_info = prepare_dataset(common_df, embolization_state)

    model, dist_matrix, feature_importances = estimate_random_forest_dissimilarity(
        X, y, holdout_eval=False, n_tree=n_tree
    )

    reduced_matrix, optimal_umap_k = perform_dimensionality_reduction(
        dist_matrix, n_neighbors=n_neighbors
    )

    bic_dir = DATA_DIR / "BIC"
    optimal_k = estimate_optimal_cluster_numbers(bic_dir)

    X_clustered, clusters = perform_clustering(
        reduced_matrix, X, X_other_info, y, optimal_k
    )

    print("Processing complete")


if __name__ == "__main__":
    main()
