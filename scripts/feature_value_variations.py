import numpy as np
import polars as pl

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)


def evaluate_feature_variations(df: pl.DataFrame) -> pl.DataFrame:
    # Calculate standard deviation for continuous features
    CONTINUOUS_DATATYPES = [
        pl.Float32,
        pl.Float64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]
    df_cont = df.clone().select(
        [
            pl.col(col)
            for col in df.columns
            if df[col].dtype in CONTINUOUS_DATATYPES or col == "state_class"
        ]
    )
    df_std = df_cont.std()
    grouped_mean_std = (
        df_cont.group_by("state_class")
        .agg(
            [
                pl.col(col).std().alias(f"{col}_std")
                for col in df_cont.columns
                if col != "state_class"
            ]
        )
        .mean()
    ).select([f"{col}_std" for col in df_cont.columns if col != "state_class"])

    df_std.select(pl.col("*").exclude("state_class")).transpose(
        include_header=True
    ).write_csv(f"./data/feature_distribution/overall_std.csv")
    grouped_mean_std.transpose(include_header=True).write_csv(
        f"./data/feature_distribution/clustered_mean_std.csv"
    )

    # Calculate entropy for categorical features
    df_cat = df.select(
        [
            pl.col(col)
            for col in df.columns
            if df[col].dtype not in CONTINUOUS_DATATYPES or col == "state_class"
        ]
    )
    overall_entropies = {}
    mean_entropies = {}
    for col in df_cat.columns:
        freq = df_cat[col].value_counts(normalize=True)["proportion"]
        overall_entropy = freq.entropy()
        overall_entropies[col] = overall_entropy

        group_entropies: list[float] = []
        df_cat_grouped = (
            df_cat.group_by("state_class")
            .agg(pl.col(col).value_counts(normalize=True).alias(f"{col}_counts"))
            .select(pl.col(f"{col}_counts"))
        )
        for row in df_cat_grouped.iter_rows():
            row_df = pl.DataFrame(row[0])
            e = row_df["proportion"].entropy()
            if e is None:
                raise ValueError(f"Entropy is None for {col}")
            group_entropies.append(e)
        mean_entropy = np.mean(group_entropies)
        mean_entropies[col] = mean_entropy

    pl.DataFrame(overall_entropies).transpose(include_header=True).write_csv(
        f"./data/overall_entropy.csv"
    )
    pl.DataFrame(mean_entropies).transpose(include_header=True).write_csv(
        f"./data/clustered_mean_entropy.csv"
    )
    return df_std


if __name__ == "__main__":
    COLUMNS = {
        "symptom": pl.Categorical,
        "SAH_severity": pl.UInt8,
        "SAH_treatment_date": pl.UInt8,
        "aneurysm_site": pl.Categorical,
        "max_diameter": pl.UInt16,
        "aneurysm_shape": pl.UInt8,
        "age": pl.Float32,
        "is_scheduled": pl.UInt8,
        "gender": pl.Categorical,
        "mRS_before": pl.UInt8,
        "responsible_physician": pl.Categorical,
        "num_superviser": pl.UInt8,
        "num_specialist": pl.UInt8,
        "num_non_specialist": pl.UInt8,
        "mRS_after": pl.UInt8,
        "info_ID": pl.Categorical,
        "treatment_ID": pl.Categorical,
        "treat_date": pl.Float32,  # pl.Date cause error in repr
        "technical_success": pl.UInt8,
        "state_class": pl.UInt16,
    }

    df_ = pl.read_csv(f"./data/state_clustered_dataframe.csv")
    CATEGORICAL_COLS = [
        key for key, value in COLUMNS.items() if value == pl.Categorical
    ]
    CAT_DICT = {col: pl.String for col in CATEGORICAL_COLS}
    df = df_.cast(CAT_DICT).cast(COLUMNS)  # type: ignore

    evaluate_feature_variations(df)
