# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
import re
from typing import List, Tuple
import eli5
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.automotive_feature_engineering.utils.utils import combine_dfs, get_feature_df
from sklearn.preprocessing import PolynomialFeatures


class FeatureInteractions:
    """
    This class handles creating polynomial feature interactions which can capture non-linear relationships
    between features and can be particularly useful in improving model performance.
    """

    def __init__(self):
        """
        Initializes the PolynomialFeatures generator to create second-degree interaction-only features without bias.
        """
        self.poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)

    ##########################################
    # Polynominal Feature Interactions
    ##########################################

    def fit_transform_polynominal_interaction_rl(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fits and transforms the DataFrame to include polynomial feature interactions based on float columns,
        excluding certain prefixed columns.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the original data.

        Returns:
        - pd.DataFrame: Transformed DataFrame including new polynomial interaction features.
        """
        float_columns = list(df.select_dtypes(include=["float64", "float32"]).columns)
        float_columns = [
            x for x in float_columns if not x.startswith(("I_", "file", "timestamp"))
        ]
        if len(float_columns) > 0:
            poly_df = pd.DataFrame(
                self.poly.fit_transform(df[float_columns]),
                columns=self.poly.get_feature_names_out(),
            )

            # Concatenate the polynomial features DataFrame with the original DataFrame
            df = combine_dfs([df.drop(float_columns, axis=1), poly_df])

        return df

    def fit_transform_polynominal_interaction_rl2(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fits and transforms the DataFrame to include polynomial feature interactions based on float columns,
        excluding certain prefixed columns.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the original data.

        Returns:
        - pd.DataFrame: Transformed DataFrame including new polynomial interaction features.
        """
        float_columns = list(df.columns)
        float_columns = [
            x for x in float_columns if not x.startswith(("I_", "file", "timestamp"))
        ]
        float_columns = list(df.select_dtypes(include=["float64", "float32"]).columns)
        float_columns = [
            x for x in float_columns if not x.startswith(("I_", "file", "timestamp"))
        ]
        if len(float_columns) > 0:
            poly_df = pd.DataFrame(
                self.poly.fit_transform(df[float_columns]),
                columns=self.poly.get_feature_names_out(),
            )

            # Concatenate the polynomial features DataFrame with the original DataFrame
            df = combine_dfs([df.drop(float_columns, axis=1), poly_df])

        return df

    def polynominal_interaction_fit(
        self, df: pd.DataFrame, target_names_list: List[str]
    ) -> object:
        """
        Fits polynomial features generator to specified numeric columns in the DataFrame, excluding columns with specific prefixes and target variables.

        Parameters:
        - df (pd.DataFrame): DataFrame containing features.
        - target_names_list (List[str]): List of column names to exclude from fitting.

        Returns:
        - Tuple[object, List[str]]: Returns a tuple containing the fitted PolynomialFeatures object and the list of column names used for fitting.
        """
        poly_regr = PolynomialFeatures(2, interaction_only=True, include_bias=False)

        float_columns = list(df.select_dtypes(include=["float64", "float32"]).columns)
        float_columns = [
            x
            for x in float_columns
            if not x.startswith(("file", "timestamp")) and x not in target_names_list
        ]
        float_columns = list(df.columns)
        float_columns = [
            x
            for x in float_columns
            if not x.startswith(("file", "timestamp")) and x not in target_names_list
        ]
        if len(float_columns) > 0:
            poly_regr.fit(df[float_columns])

        return poly_regr, float_columns

    def polynominal_interaction_transform(
        self,
        df: pd.DataFrame,
        poly_regr: object = None,
        float_columns: list[str] = None,
    ) -> pd.DataFrame:
        """
        Transforms the DataFrame by applying previously fitted polynomial features, adding interaction terms to the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame to transform.
        - poly_regr (object): Fitted PolynomialFeatures object.
        - float_columns (List[str]): List of column names on which to apply the transformation.

        Returns:
        - pd.DataFrame: DataFrame with additional polynomial interaction features.
        """
        if len(float_columns) > 0:
            poly_df = pd.DataFrame(
                poly_regr.transform(df[float_columns]),
                columns=poly_regr.get_feature_names_out(),
            )

            # Concatenate the polynomial features DataFrame with the original DataFrame
            df = combine_dfs([df.drop(float_columns, axis=1), poly_df])

        return df

    ##########################################
    # Erstellen eines Zeitfensters
    ##########################################
    def fit_transform_make_windowed(self, df: pd.DataFrame, windowsize: int):
        """
        Creates windowed features for time series forecasting, by shifting columns over a specified window size.

        Parameters:
        - df (pd.DataFrame): DataFrame containing time series data.
        - windowsize (int): Number of time steps to include in the window.

        Returns:
        - pd.DataFrame: DataFrame with windowed features.
        """
        columns = df.columns
        shifted_dfs = []

        for col in columns:
            shifted_cols = [f"t-{i}_{col}" for i in range(windowsize, 0, -1)]
            shifted_dfs.append(
                pd.concat([df[col].shift(i) for i in range(windowsize, 0, -1)], axis=1)
            )
            shifted_dfs[-1].columns = shifted_cols

        df = pd.concat([df] + shifted_dfs, axis=1)

        return df
