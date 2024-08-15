# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
import re
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from src.automotive_feature_engineering.utils.utils import combine_dfs


class FeatureEncoding:
    """
    This class provides methods for encoding categorical features within a dataset.
    The primary method of encoding used is One-Hot Encoding, which transforms categorical
    variable(s) into a binary matrix representation.
    """

    ##########################################
    # One-Hot-Encoding
    ##########################################

    def one_hot_encoding_fit(self, df: pd.DataFrame) -> object:
        """
        Fits a OneHotEncoder to the DataFrame for categorical features.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data with categorical variables.

        Returns:
        - object: Fitted OneHotEncoder object that can be used to transform datasets.
        """
        ohe_regr = OneHotEncoder(handle_unknown="ignore", dtype=int)
        string_columns = list(df.select_dtypes(include=["object", "category"]).columns)
        ohe_regr.fit(df[string_columns].astype(str))
        return ohe_regr

    def one_hot_encoding_transform(
        self, df: pd.DataFrame, ohe_regr: object = None
    ) -> pd.DataFrame:
        """
        Transforms the DataFrame's categorical columns using the provided OneHotEncoder.

        Parameters:
        - df (pd.DataFrame): DataFrame to transform.
        - ohe_regr (object): Fitted OneHotEncoder object.

        Returns:
        - pd.DataFrame: Transformed DataFrame with original categorical features replaced by their one-hot encoded counterparts.
        """
        # transform string columns
        # feature_names_df.difference(feature_names_ohe)
        diff = list(
            set(ohe_regr.feature_names_in_)
            - set(list(df.select_dtypes(include=["object", "category"]).columns))
        )
        if len(diff) > 0:
            df_test = pd.DataFrame(columns=np.array(diff))
            df_test.fillna(value=0, inplace=True)
            df = pd.concat([df, df_test], axis="columns")

        # string_columns = list(df.select_dtypes(include=["object", "category"]).columns)
        # string_columns.sort()

        # df = df.reindex(sorted(df.columns), axis=1)

        df_out = ohe_regr.transform(df[ohe_regr.feature_names_in_].astype(str))
        df_out = df_out.toarray()

        # get column names
        df_out = pd.DataFrame(df_out, columns=ohe_regr.get_feature_names_out())
        df = df.drop(ohe_regr.feature_names_in_, axis=1)

        # append to original dataframe
        df_out.reset_index(drop=True, inplace=True)
        df.reset_index(inplace=True)

        df = combine_dfs([df_out, df])

        return df
