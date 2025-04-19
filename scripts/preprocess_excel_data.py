import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.neighbors._base

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from missingpy import MissForest

DATA_DIR = Path("./data")

renamer = {
    "登録番号": "registration_number",
    "基本情報ID": "info_ID",
    "データID": "data_ID",
    "治療責任医師": "responsible_physician",
    "治療施設": "facility",
    "治療年月日": "treatment_date",
    "スクラブイン関与人数 指導医": "num_superviser",
    "専門医": "num_specialist",
    "非専門医": "num_non_specialist",
    "予定または緊急": "is_scheduled",
    "治療時年齢": "age",
    "性別": "gender",
    "治療対象疾患名": "disease",
    "実施治療法": "treatment",
    "技術的成功(technical success)": "technical_success",
    "麻酔": "anesthesia",
    "同時他治療数": "num_other_treatments",
    "治療の合併症": "complication",
    "合併症の詳細": "complication_detail",
    "合併症の転帰": "complication_outcome",
    "発症前mRS(Grade)": "mRS_before",
    "治療30日後のmRS(Grade)": "mRS_after",
    "有害事象の発生": "adverse_event",
    "有害事象の詳細": "adverse_event_detail",
    "有害事象の治療との関連性": "adverse_event_relevance",
    "治療番号": "treatment_ID",
    "症候": "symptom",
    "SAH重症度(WFNS　Grade)": "SAH_severity",
    "SAH治療日": "SAH_treatment_date",
    "最大径（mm）": "max_diameter",
    "形状": "aneurysm_shape",
    "部位": "aneurysm_site",
    "血管撮影装置": "angiography_device",
    "3D回転血管撮影装置": "3D_angiography_device",
    "治療戦略": "strategy",
    "使用コイル": "coil",
    "塞栓結果": "result",
    "術中ヘパリン使用": "heparin",
    "術中ヘパリン使用のタイミング": "heparin_timing",
    "術前抗血小板薬の使用": "antiplatelet_before",
    "術前抗血小板薬の詳細": "antiplatelet_before_detail",
    "術後抗血小板薬の使用": "antiplatelet_after",
    "術後抗血小板薬の詳細": "antiplatelet_after_detail",
    "術後抗血栓療法の実施": "antithrombotic_therapy",
    "術後抗血栓療法の詳細": "antithorombotic_therapy_detail",
}


def convert2onehot_encoding(df, column_name: str, num_factor: int):
    OHE = pd.DataFrame(
        np.zeros((df.shape[0], num_factor)),
        columns=[f"{column_name}_{i}" for i in range(num_factor)],
        dtype=np.uint8,
    )

    multi_disease_idx = df[column_name].apply(type) == str
    for idx, value in df.loc[multi_disease_idx, column_name].items():
        values = value.split(",")
        for v in values:
            OHE.loc[idx, f"{column_name}_{v}"] = 1
    # not multi values row
    for idx, value in df.loc[~multi_disease_idx, column_name].items():
        OHE.loc[idx, f"{column_name}_{value}"] = 1

    return OHE


if __name__ == "__main__":
    ## 治療情報
    treat_df = pd.read_excel(
        DATA_DIR
        / "JRNET3_DATA_20210929_脳神経血管内治療学会より/00_治療情報_解析用.xlsx"
    )
    treat_df = treat_df.drop(["治療番号", "治療対象疾患名.1"], axis=1)
    treat_df = treat_df.rename(renamer, axis=1)
    treat_df.loc[treat_df.complication_outcome.isna(), "complication_outcome"] = 0

    for column_name, num_factor in [
        ("disease", 17),
        ("anesthesia", 6),
        ("complication_detail", 8),
    ]:
        OHE = convert2onehot_encoding(treat_df, column_name, num_factor)
        treat_df = treat_df.drop(column_name, axis=1)
        treat_df = pd.concat([treat_df, OHE], axis=1)

    ##  症例基本情報
    basic_df = pd.read_excel(
        DATA_DIR
        / "JRNET3_DATA_20210929_脳神経血管内治療学会より/00_症例基本情報_解析用.xlsx"
    )
    basic_df = basic_df.rename(renamer, axis=1)

    ## 共通調査項目
    common_df = pd.merge(
        basic_df.drop("registration_number", axis=1),
        treat_df.drop("registration_number", axis=1),
        on="info_ID",
    )
    common_df = common_df.reindex(
        [
            "info_ID",
            "data_ID",
            "treat_date",
            "age",
            "facility",
            "is_scheduled",
            "num_other_treatments",
            "gender",
            "mRS_before",
            "mRS_after",
            "adverse_event",
            "adverse_event_detail",
            "adverse_event_relevance",
            "responsible_physician",
            "num_superviser",
            "num_specialist",
            "num_non_specialist",
            "treatment",
            "technical_success",
            "complication",
            "complication_outcome",
            "disease_0",
            "disease_1",
            "disease_2",
            "disease_3",
            "disease_4",
            "disease_5",
            "disease_6",
            "disease_7",
            "disease_8",
            "disease_9",
            "disease_10",
            "disease_11",
            "disease_12",
            "disease_13",
            "disease_14",
            "disease_15",
            "disease_16",
            "anesthesia_0",
            "anesthesia_1",
            "anesthesia_2",
            "anesthesia_3",
            "anesthesia_4",
            "anesthesia_5",
            "complication_detail_0",
            "complication_detail_1",
            "complication_detail_2",
            "complication_detail_3",
            "complication_detail_4",
            "complication_detail_5",
            "complication_detail_6",
            "complication_detail_7",
        ],
        axis=1,
    )
    common_df.to_csv(DATA_DIR / "common_preprocessed.csv", index=False)

    ## impute via missForest
    common_df.treat_date = common_df.treat_date.apply(lambda x: int(x.timestamp()))
    common_df.treat_date = common_df.treat_date.astype(np.uint32)

    imputer = MissForest(n_estimators=100, n_jobs=-1)
    imputed_array = imputer.fit_transform(common_df, cat_vars=[4, 5, 7, 14, 15, 16])
    imputed_df = pd.DataFrame(imputed_array, columns=common_df.columns)
    imputed_df.to_csv(DATA_DIR / "common_imputed.csv", index=False)

    ## Embolization specific data
    embolization_df = pd.read_excel(
        DATA_DIR
        / "JRNET3_DATA_20210929_脳神経血管内治療学会より/01_脳動脈瘤塞栓術（瘤内塞栓術、母血管温存）情報 のコピー.xlsx"
    )
    embolization_df = embolization_df.rename(renamer, axis=1)
    embolization_state = embolization_df.loc[:, "data_ID":"3D_angiography_device"]
    embolization_state.to_csv(DATA_DIR / "embolization_state.csv", index=False)

    embolization_action = embolization_df.loc[
        :, "angiography_device":"antiplatelet_after_detail"
    ]
    embolization_action = embolization_action.drop("result", axis=1)
    for column, num_factor in (
        ("strategy", 8),
        ("coil", 6),
        ("antiplatelet_before_detail", 7),
        ("antiplatelet_after_detail", 7),
        ("antithorombotic_therapy_detail", 6),
    ):
        OHE = convert2onehot_encoding(embolization_action, column, num_factor)
        embolization_action = embolization_action.drop(column, axis=1)
        embolization_action = pd.concat([embolization_action, OHE], axis=1)

    embolization_action.to_csv(DATA_DIR / "embolization_action.csv", index=False)
