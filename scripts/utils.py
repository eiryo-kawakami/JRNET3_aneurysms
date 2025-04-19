import json
import math
import re
import sqlite3
import sys
from pprint import pprint
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dbname = "feature_names.db"
db_path = "./data/" + dbname


def register_renamer(renamer: dict):
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()
    for key, value in renamer.items():
        curs.execute(
            "INSERT into english_features (ja_features,en_features) VALUES(?,?)",
            (key, value),
        )
    conn.commit()
    conn.close()


def modify_renamer(renamer: dict):
    new_renamer = {}
    for key, value in renamer.items():
        new_renamer[key] = {}
        new_renamer[key]["en"] = value
        for i in range(18):
            new_renamer[key][i] = ""
    pprint(new_renamer)


def register_features_details():
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()
    with open("./modified_renamer.json", "rt") as f:
        detailed_renamer = json.load(f)

    for key, detail_dict in detailed_renamer.items():
        curs.execute("SELECT * FROM english_features WHERE ja_features=?", (key,))
        feature_info = curs.fetchall()[0]
        feature_id = feature_info[0]

        for detail_id, value in detail_dict.items():
            if detail_id == "en":
                continue
            curs.execute(
                "INSERT INTO features_details (feature_id,detail_id,detail_ja) VALUES(?,?,?)",
                (feature_id, detail_id, value),
            )

    conn.commit()
    conn.close()


def full2half(string: str):
    trans = {
        "（": "(",
        "）": ")",
    }
    string = string.translate(str.maketrans(trans))
    string = re.sub(r"\s", " ", string)
    return string


def get_ja_feature_expression(en_detailed_feature: str):
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    detail_id_match = re.search(r"(?<=_)\d+", en_detailed_feature)
    if detail_id_match:
        en_feature = en_detailed_feature[: detail_id_match.start() - 1]
        detail_id = detail_id_match.group()
    else:
        en_feature = en_detailed_feature
        detail_id = None

    if en_feature == "cluster":
        return "cluster"
    feature_id, ja_feature = curs.execute(
        "SELECT id,ja_features FROM english_features WHERE en_features=?", (en_feature,)
    ).fetchall()[0]
    if detail_id_match:
        # 定義において detail_id が連番になっていない
        if (en_feature == "SAH_severity") and (detail_id == "6"):
            detail_id = 9
        try:
            detail_ja = curs.execute(
                "SELECT detail_ja FROM features_details WHERE feature_id=? AND detail_id=?",
                (feature_id, detail_id),
            ).fetchall()[0][0]
        except IndexError:
            print(en_feature, detail_id)
            sys.exit()
    else:
        return ja_feature

    ja_feature_expression = full2half(ja_feature) + ":" + full2half(detail_ja)
    return ja_feature_expression


def get_en_renamer() -> dict:
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()
    results = curs.execute(
        "SELECT ja_features,en_features FROM english_features"
    ).fetchall()

    renamer = {}
    for ja, en in results:
        renamer[ja] = en
    return renamer


def convert2onehot_encoding(
    df, column_name: str, num_factor: int, factor_start_value=0
):
    OHE = pd.DataFrame(
        np.zeros((df.shape[0], num_factor)),
        columns=[
            f"{column_name}_{i}"
            for i in range(factor_start_value, factor_start_value + num_factor)
        ],
        dtype=np.uint8,
    )
    OHE.index = df.index

    if df[column_name].dtype == "object":
        single_value_indice = ~df[column_name].str.contains(",")
    else:
        single_value_indice = pd.Series(np.ones((df.shape[0],)), dtype=bool)
        df[column_name] = df[column_name].astype(np.uint8)

    for idx, value in df.loc[~single_value_indice, column_name].items():
        values = value.split(",")
        for v in values:
            OHE.loc[idx, f"{column_name}_{v}"] = 1

    # not multi values row
    for idx, value in df.loc[single_value_indice, column_name].items():
        value = int(value)
        OHE.loc[idx, f"{column_name}_{value}"] = 1

    return OHE


def convert_columns_OHE(
    df: pd.DataFrame, column_num_factor: dict[str, Union[int, list]]
):
    for column_name, num_factor in column_num_factor.items():
        if isinstance(num_factor, list):
            OHE = convert2onehot_encoding(df, column_name, num_factor[0], num_factor[1])
        else:
            OHE = convert2onehot_encoding(df, column_name, num_factor)
        df = df.drop(column_name, axis=1)
        df = pd.concat([df, OHE], axis=1)
    return df


def L_method(loss_array):
    def rmse(A, y):
        resid = np.linalg.lstsq(A, y, rcond=None)[1]
        if not resid.size > 0:
            resid = 0
        return math.sqrt(resid / y.shape[0])

    def get_knee(cutoff):
        rmse_list = []
        for c in range(2, cutoff - 1):
            y_L = loss_array[:c]
            y_R = loss_array[c:cutoff]
            A_L = np.vstack([range(c), np.ones(c)]).T
            A_R = np.vstack([range(c, cutoff), np.ones(cutoff - c)]).T

            rmse_L = rmse(A_L, y_L)
            rmse_R = rmse(A_R, y_R)

            rmse_overall = (c * rmse_L + (cutoff - c) * rmse_R) / cutoff
            rmse_list.append(rmse_overall)
        return np.argmin(rmse_list) + 2  # as c starts from 2

    cutoff_ = loss_array.shape[0]
    current_knee = cutoff_

    while True:
        last_knee = current_knee
        current_knee = get_knee(cutoff_)
        if current_knee >= last_knee or current_knee <= 10:
            print("knee:", current_knee + 2)
            fig, ax = plt.subplots(1, 1, figsize=(12, 3))
            ax.scatter(range(2, cutoff_ + 2), loss_array[-cutoff_:], s=3)
            ax.set_xticks(range(2, cutoff_ + 2))
            return current_knee + 2
            break
        cutoff_ = current_knee * 2


state_columns = [
    "symptom",
    "SAH_severity",
    "SAH_treatment_date",
    "aneurysm_site",
    "max_diameter",
    "aneurysm_shape",
    "age",
    "is_scheduled",
    "gender",
    "mRS_before",
    "responsible_physician",
    "num_superviser",
    "num_specialist",
    "num_non_specialist",
    "disease_1",
    "disease_2",
    "disease_3",
    "disease_5",
    "disease_6",
    "disease_7",
    "disease_8",
    "disease_9",
    "disease_12",
    "disease_13",
    "disease_15",
    "disease_16",
    "treat_date",
]

action_useless_columns = [
    "strategy_0",
    "strategy_6",
    "strategy_7",
    "coil_0",
    "coil_4",
    "coil_5",
    "antiplatelet_before_detail_5",
    "antiplatelet_before_detail_6",
    "antiplatelet_after_detail_5",
    "antiplatelet_after_detail_6",
    "antithorombotic_therapy_detail_4",
    "antithorombotic_therapy_detail_5",
    "angiography_device_0",
    "angiography_device_3",
    "3D_angiography_device_0",
    "3D_angiography_device_3",
    "heparin_0",
    "heparin_3",
    "heparin_timing_0",
    "heparin_timing_4",
    "heparin_timing_5",
    "antiplatelet_before_0",
    "antiplatelet_before_3",
    "antiplatelet_after_0",
    "antiplatelet_after_3",
    "antithrombotic_therapy_0",
    "antithrombotic_therapy_3",
]
