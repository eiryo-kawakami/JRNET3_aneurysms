from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

state_cols = [v for v in utils.state_columns if "disease_" not in v]

DATA_DIR = Path("./data")


def preprocess():
    state_X = pd.read_csv(DATA_DIR / "state_clustered_dataframe.csv")

    state_X.mRS_before = state_X.mRS_before - 1
    state_X.mRS_after = state_X.mRS_after - 1
    state_X["mRS_diff"] = state_X.mRS_after - state_X.mRS_before

    action_X = pd.read_csv(DATA_DIR / "URF/action_clustered_dataframe.csv")
    action_X["action_class"] = action_X["cluster"].astype(np.uint8)
    action_X = action_X.drop("cluster", axis=1)

    X = pd.merge(action_X, state_X, how="inner", on=["info_ID", "treatment_ID"])
    X["mRS_diff"] = X["mRS_diff"].astype(np.int8)
    num_states = np.unique(state_X["state_class"]).shape[0]
    return X, state_X, num_states


def save_action_mRS_diff(X: pd.DataFrame):
    action_cols = X.columns[:31].tolist()
    action_mrs_diff = X[
        action_cols
        + [
            "antiplatelet_before_1",
            "antiplatelet_after_1",
            "antithrombotic_therapy_1",
            "state_class",
            "mRS_diff",
        ]
    ]
    remove_cols = [
        "antiplatelet_before_detail_0",
        "antiplatelet_after_detail_0",
        "antithorombotic_therapy_detail_0",
        "3D_angiography_device_1",
    ]
    action_mrs_diff = action_mrs_diff.drop(remove_cols, axis=1)

    for i in tqdm(X.state_class.unique()):
        action_mrs_diff.loc[action_mrs_diff.state_class == i].drop(
            "state_class", axis=1
        ).to_csv(DATA_DIR / f"action_features_mRS_diff/state_{i}.csv", index=False)


def aggregate_p_values(num_states):
    """Read all the p values from the csv file and aggregate them into a dataframe

    Notes:
        p values are calculated in `multiple_exact_wicox_test.R`.
        Run the R script before running this function.
    """
    p_values = []
    for i in range(num_states):
        sr = pd.read_csv(
            DATA_DIR / f"action_features_mRS_diff/wilcox/state_{i}.csv",
            index_col=0,
        )["p_value"]
        significant_features = sr[sr < 0.05]
        if significant_features.shape[0] > 0:
            for feature in significant_features.index:
                row = [i, feature, significant_features[feature]]
                p_values.append(row)

    p_values = pd.DataFrame(p_values, columns=["state", "feature", "p_value"])
    p_values.to_csv(
        DATA_DIR / "significant_action_features/wilcox_p_values.csv", index=False
    )
    p_values.feature.value_counts().to_csv(
        DATA_DIR / "significant_action_features/wilcox_p_values_count.csv"
    )
    return p_values


def one_feature_comparison(X: pd.DataFrame, p_values: pd.DataFrame, feature_name: str):
    """Compare the mRS difference between those treated with the feature and those not treated with it."""
    feature_outperform_states = []
    underperform_states = []
    for i, (state, p) in p_values.loc[
        p_values["feature"] == feature_name, ["state", "p_value"]
    ].iterrows():
        # antiplatelet_after_1=1 means antiplatelet is not used
        state_group_mean = (
            X.loc[X["state_class"] == state, [feature_name, "mRS_diff"]]
            .groupby(feature_name)
            .median()
        )
        applied_mRS_diff = state_group_mean.at[1, "mRS_diff"]
        not_applied_mRS_diff = state_group_mean.at[0, "mRS_diff"]
        # 0 means antiplatelet_after is used
        feature_advantage = applied_mRS_diff - not_applied_mRS_diff
        # if the mRS diff for the group using the feature is smaller than the other group, then the feature outperforms
        if feature_name in ("antiplatelet_after_1", "antithrombotic_therapy_1"):
            if feature_advantage > 0:
                feature_outperform_states.append(state)
            elif feature_advantage < 0:
                underperform_states.append(state)
            else:
                print(f"state {state} is neutral")
        else:
            if feature_advantage < 0:
                feature_outperform_states.append(state)
            elif feature_advantage > 0:
                underperform_states.append(state)
            else:
                print(f"state {state} is neutral")

    print(f"number of outperform states: {len(feature_outperform_states)}")
    print(feature_outperform_states)
    print(f"number of underperform states: {len(underperform_states)}")
    print(underperform_states)

    df = X.copy(deep=True)
    new_col_name = "outperform"
    df[new_col_name] = 0
    df.loc[df["state_class"].isin(feature_outperform_states), new_col_name] = 1
    df.loc[df["state_class"].isin(underperform_states), new_col_name] = -1

    y = df[new_col_name]
    df.to_csv(DATA_DIR / f"significant_action_features/{feature_name}.csv", index=False)

    return df, y


def significant_states_plot(X: pd.DataFrame, feature_name: str):
    """Bubble plot of the states with significant differences in mRS scores.

    Args:
        X: output dataframe from `one_feature_comparison` function
    """

    state_df = pd.read_csv(DATA_DIR / "state_clustered_dataframe.csv")
    scatter_matrix = np.load(DATA_DIR / "state_scatter_matrix.npz")["arr_0"]

    outperform_states = X[X.outperform == 1].state_class.unique()
    underperform_states = X[X.outperform == -1].state_class.unique()
    nonsignificant_states = X[X.outperform == 0].state_class.unique()

    cluster_size = state_df.groupby("state_class").size()
    cluster_centers = []
    for state in cluster_size.index:
        state_indice = state_df[state_df.state_class == state].index.tolist()
        cluster_centers.append(scatter_matrix[state_indice].mean(axis=0))
    cluster_centers = np.array(cluster_centers)

    ax_size = 6
    fig, ax = plt.subplots(1, 1, figsize=(ax_size * 1, ax_size * 1))
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    def bubble_plot(cluster_indice, ax, alpha_: float, color: str, label_: str):
        scatter = ax.scatter(
            x=cluster_centers[cluster_indice, 0],
            y=cluster_centers[cluster_indice, 1],
            s=cluster_size[cluster_indice] * 5,
            alpha=alpha_,
            c=color,
            label=label_,
        )

    bubble_plot(nonsignificant_states, ax, 0.2, "gray", "nonsignificant")
    bubble_plot(outperform_states, ax, 0.9, "crimson", "outperform")
    bubble_plot(underperform_states, ax, 0.9, "royalblue", "underperform")

    ax.legend()
    dict_features = {
        "antiplatelet_after_1": "Antiplatelet agent after surgery",
        "antithrombotic_therapy_1": "Antithrombotic therapy",
        "antiplatelet_after_detail_3": "Cilostazol",
    }
    if feature_name in dict_features:
        ax.set_title(dict_features[feature_name], fontsize=14)
    fig.savefig(DATA_DIR / f"../fig/{feature_name}_significant_states.pdf")


def calculate_feature_ratios(
    X: pd.DataFrame, p_values: pd.DataFrame, feature_name: str
):
    """Returns the ratio of patients who were treated with the feature.

    Returns:
    - overall_ratio (float): The ratio of patients who were treated with the given feature.
    - states_ratio (float): The ratio of patients who were treated with the given feature,
        belonging to suboptimal treatment states.
    """
    overall_ratio = X[feature_name].mean()
    significant_states = p_values[p_values.feature == feature_name].state.unique()
    states_ratio = X[X.state_class.isin(significant_states)][feature_name].mean()
    if feature_name in (
        "antiplatelet_after_1",
        "antiplatelet_before_1",
        "antithrombotic_therapy_1",
    ):
        overall_ratio = 1 - overall_ratio
        states_ratio = 1 - states_ratio
    return overall_ratio, states_ratio


def main():
    X, state_X, num_states = preprocess()
    p_values = aggregate_p_values(num_states)
    print(p_values.feature.value_counts())
    print(p_values.feature.value_counts().sum())

    # Set feature names based on p-values
    feature_names_ = [
        "antiplatelet_after_1",
        "antiplatelet_after_detail_3",
        "antithrombotic_therapy_1",
        "antithorombotic_therapy_detail_3",
        "heparin_timing_2",
        "angiography_device_1",
        "angiography_device_2",
        "antiplatelet_before_1",
    ]

    for feature_name in feature_names_:
        df, y = one_feature_comparison(X, p_values, feature_name)
        significant_states_plot(df, feature_name)


if __name__ == "__main__":
    main()
