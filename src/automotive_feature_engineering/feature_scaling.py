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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import preprocessing


class FeatureScaling:
    """
    A class for scaling features using different scaling methods. Scaling is crucial for algorithms that
    are sensitive to the magnitude of inputs like gradient descent-based algorithms, k-nearest neighbors,
    and support vector machines.
    """

    ##########################################
    # min-max scaling
    ##########################################

    def minmax_scaler_fit(self, df: pd.DataFrame) -> object:
        """
        Fits a MinMaxScaler to the DataFrame, scaling features to a range.

        Parameters:
        df (pd.DataFrame): The DataFrame whose features are to be scaled.

        Returns:
        MinMaxScaler: A fitted scaler instance which can be used to transform data.
        """
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(df)

        return minmax_scaler

    def minmax_scaler_transform(
        self, df: pd.DataFrame, minmax_scaler: object
    ) -> pd.DataFrame:
        """
        Transforms the DataFrame using a previously fitted MinMaxScaler.

        Parameters:
        df (pd.DataFrame): The DataFrame to transform.
        minmax_scaler (MinMaxScaler): A fitted MinMaxScaler instance.

        Returns:
        pd.DataFrame: The transformed DataFrame with features scaled to the range [0, 1].
        """
        transformed_df = minmax_scaler.transform(df)
        df = pd.DataFrame(transformed_df, columns=df.columns)

        return df

    ##########################################
    # standard scaling z = (x - u) / s
    ##########################################

    def standard_scaler_fit(self, df: pd.DataFrame, target_names: List) -> object:
        """
        Fits a StandardScaler to the DataFrame, standardizing features by removing the mean and scaling to unit variance.

        Parameters:
        df (pd.DataFrame): The DataFrame whose features are to be standardized.

        Returns:
        StandardScaler: A fitted scaler instance which can be used to transform data.
        """
        standard_scaler = StandardScaler()
        standard_scaler.fit(df)

        return standard_scaler

    def standard_scaler_transform(
        self, df: pd.DataFrame, standard_scaler: object
    ) -> pd.DataFrame:
        """
        Transforms the DataFrame using a previously fitted StandardScaler.

        Parameters:
        df (pd.DataFrame): The DataFrame to transform.
        standard_scaler (StandardScaler): A fitted StandardScaler instance.

        Returns:
        pd.DataFrame: The transformed DataFrame with standardized features.
        """
        transformed_df = standard_scaler.transform(df)
        df = pd.DataFrame(transformed_df, columns=df.columns)

        return df

    ##########################################
    # robust scaling
    ##########################################
    def robust_scaler_fit(self, df: pd.DataFrame) -> object:
        """
        Fits a RobustScaler to the DataFrame, scaling features using statistics that are robust to outliers.

        Parameters:
        df (pd.DataFrame): The DataFrame whose features are to be robustly scaled.

        Returns:
        RobustScaler: A fitted scaler instance which can be used to transform data.
        """
        robust_scaler = RobustScaler()
        robust_scaler.fit(df)

        return robust_scaler

    def robust_scaler_transform(
        self, df: pd.DataFrame, robust_scaler: object
    ) -> pd.DataFrame:
        """
        Transforms the DataFrame using a previously fitted RobustScaler.

        Parameters:
        df (pd.DataFrame): The DataFrame to transform.
        robust_scaler (RobustScaler): A fitted RobustScaler instance.

        Returns:
        pd.DataFrame: The transformed DataFrame with features scaled using robust statistics.
        """
        transformed_df = robust_scaler.transform(df)
        df = pd.DataFrame(transformed_df, columns=df.columns)

        return df
