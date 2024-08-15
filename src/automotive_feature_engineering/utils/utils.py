# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from typing import Any, Dict, Optional
from typing import List
import pandas as pd
import os
import json


def rename_logfile(logger, doku, fuses):
    os.rename(
        str(logger.handlers[0].baseFilename),
        doku.get_path()
        + "/"
        + ("_".join(str(x) for x in fuses)[0:15])
        + "-EnergAIze.log",
    )


def sort_df_columns_by_name(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(sorted(df.columns), axis=1)


def remove_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]


def combine_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    dfs_modified = [df.reset_index(drop=True) for df in dfs]
    combined_df = remove_duplicate_cols(pd.concat(dfs_modified, axis=1))
    combined_df.index = dfs[0].index
    return combined_df


def get_feature_df(
    df: pd.DataFrame, target_names_list: List[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(target_names_list) > 0:
        target_df = df[target_names_list]
        feature_df = df.drop(columns=target_names_list, axis=1)
    else:
        target_df = pd.DataFrame()
        feature_df = df

    return feature_df, target_df


# Split df for multiprocessing
def split_df(df: pd.DataFrame) -> pd.DataFrame:
    # split the df
    num_cols_per_split = 200
    split_dfs = [
        df.iloc[:, i : i + num_cols_per_split]
        for i in range(0, len(df.columns), num_cols_per_split)
    ]

    return split_dfs


def get_dict(
    config_path: Optional[str],
    default_path: str,
) -> Dict[str, Any]:
    print(f"Default path: {default_path}")
    assert os.path.isfile(default_path)
    with open(default_path) as default_file:
        default_dict = json.load(default_file)
        if not config_path:
            return default_dict

        assert os.path.isfile(config_path)
        with open(config_path) as config_file:
            config_dict = json.load(config_file)

            return default_dict | config_dict
