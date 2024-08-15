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
import warnings

import multiprocessing as mp

# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
import time
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from pandas.api.types import is_string_dtype, is_object_dtype
from src.automotive_feature_engineering.utils.utils import split_df, combine_dfs


class SnaHandling:
    def _is_float(self, x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    def fill_sna_median_fit(self, df_train: pd.DataFrame, th: int) -> dict:
        """
        Calculate and return a dictionary mapping each column to its median value if it has sufficient unique values above a threshold 'th'; otherwise, map to 'sna'.

        Parameters:
        df_train (pd.DataFrame): The training dataset.
        th (int): Threshold specifying the minimum number of unique values a column must have to compute its median.

        Returns:
        dict: A dictionary where each value is the computed median or 'sna'.
        """
        sna_dict = {}
        for col in df_train.columns:
            if len(df_train[col].value_counts()) >= th:
                col_numeric = pd.to_numeric(df_train[col], errors="coerce")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    median = col_numeric.median(skipna=True)

                if self._is_float(median) and not pd.isnull(median):
                    sna_dict[col] = median
                else:
                    sna_dict[col] = "sna"
            else:
                sna_dict[col] = "sna"

        return sna_dict

    def fill_sna_median_transform(
        self, df: pd.DataFrame, sna_dict: dict
    ) -> pd.DataFrame:
        """
        Transform the dataframe by replacing NaNs according to a given dictionary `sna_dict` that contains median values or 'sna' for columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be transformed.
        sna_dict (dict): A dictionary containing median values or 'sna' as fallback values for the columns in `df`.

        Returns:
        pd.DataFrame: The transformed dataframe with NaNs filled based on `sna_dict`.
        """
        for col in df.columns:
            try:
                if self._is_float(sna_dict[col]):
                    df[col] = [
                        (
                            float(val)
                            if self._is_float(val)
                            else sna_dict[col] if "sna" in str(val).lower() else 0
                        )
                        for val in df[col]
                    ]
                    df[col].fillna(sna_dict[col], inplace=True)
                else:
                    df[col] = [
                        sna_dict[col] if "sna" in val.lower() else str(val)
                        for val in df[col].astype(str)
                    ]
                    df[col] = df[col].astype(str).fillna("nan")
            except KeyError:
                continue

        return df

    def fill_sna_mean_fit(self, df_train: pd.DataFrame, th: int) -> dict:
        """
        Calculate and return a dictionary mapping each column to its mean value if it has sufficient unique values above a threshold 'th'; otherwise, map to 'sna'.

        Parameters:
        df_train (pd.DataFrame): The training dataset.
        th (int): Threshold specifying the minimum number of unique values a column must have to compute its median.

        Returns:
        dict: A dictionary where each value is the computed mean or 'sna'.
        """
        sna_dict = {}
        for col in df_train.columns:
            if len(df_train[col].value_counts()) >= th:
                col_numeric = pd.to_numeric(df_train[col], errors="coerce")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    mean = col_numeric.mean(skipna=True)

                if self._is_float(mean) and not pd.isnull(mean):
                    sna_dict[col] = mean
                else:
                    sna_dict[col] = "sna"
            else:
                sna_dict[col] = "sna"

        return sna_dict

    def fill_sna_mean_transform(self, df: pd.DataFrame, sna_dict: dict) -> pd.DataFrame:
        """
        Transform the dataframe by replacing NaNs according to a given dictionary `sna_dict` that contains mean values or 'sna' for columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be transformed.
        sna_dict (dict): A dictionary containing median values or 'sna' as fallback values for the columns in `df`.

        Returns:
        pd.DataFrame: The transformed dataframe with NaNs filled based on `sna_dict`.
        """
        for col in df.columns:
            if self._is_float(sna_dict[col]):
                df[col] = [
                    (
                        float(val)
                        if self._is_float(val)
                        else sna_dict[col] if "sna" in str(val).lower() else 0
                    )
                    for val in df[col]
                ]
                df[col].fillna(sna_dict[col], inplace=True)
            else:
                df[col] = [
                    sna_dict[col] if "sna" in val.lower() else str(val)
                    for val in df[col].astype(str)
                ]
                df[col] = df[col].astype(str).fillna("nan")

        return df

    def fill_sna_zero_fit(self, df_train: pd.DataFrame, th: int) -> dict:
        """
        Calculate and return a dictionary mapping each column to 0 if it has sufficient unique values above a threshold 'th'; otherwise, map to 'sna'.

        Parameters:
        df_train (pd.DataFrame): The training dataset.
        th (int): Threshold specifying the minimum number of unique values a column must have to compute its median.

        Returns:
        dict: A dictionary where each column is mapped to 0 if the uniqueness condition is met, otherwise to 'sna'.
        """
        sna_dict = {}
        for col in df_train.columns:
            if len(df_train[col].value_counts()) >= th:
                col_numeric = pd.to_numeric(df_train[col], errors="coerce")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    mean = col_numeric.mean(skipna=True)

                if self._is_float(mean) and not pd.isnull(mean):
                    sna_dict[col] = 0
                else:
                    sna_dict[col] = "sna"
            else:
                sna_dict[col] = "sna"

        return sna_dict

    def fill_sna_zero_transform(self, df: pd.DataFrame, sna_dict: dict) -> pd.DataFrame:
        """
        Transform the dataframe by replacing NaNs according to a given dictionary `sna_dict` that contains 0 or 'sna' for columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be transformed.
        sna_dict (dict): A dictionary containing 0 or 'sna' as fallback values for the columns in `df`.

        Returns:
        pd.DataFrame: The transformed dataframe with NaNs filled based on `sna_dict`.
        """
        for col in df.columns:
            if self._is_float(sna_dict[col]):
                df[col] = [
                    (
                        float(val)
                        if self._is_float(val)
                        else sna_dict[col] if "sna" in str(val).lower() else 0
                    )
                    for val in df[col]
                ]
                df[col].fillna(sna_dict[col], inplace=True)
            else:
                df[col] = [
                    sna_dict[col] if "sna" in val.lower() else str(val)
                    for val in df[col].astype(str)
                ]
                df[col] = df[col].astype(str).fillna("nan")
        return df

    def fill_sna_median_rl(
        self, df_train: pd.DataFrame, th: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fills missing values in the DataFrame using the median of columns where applicable, based on a threshold for unique values.

        Parameters:
        df (pd.DataFrame): The DataFrame to be cleaned.
        df_train (pd.DataFrame): Training data DataFrame used as a reference to calculate the median.
        th (int): Threshold number of unique values required to consider a column for median filling.

        Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the updated main DataFrame and training DataFrame.
        """
        for col in df_train.columns:
            if len(df_train[col].value_counts()) >= th:
                col_numeric = pd.to_numeric(df_train[col], errors="coerce")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    median = col_numeric.median(skipna=True)

                if self._is_float(median) and not pd.isnull(median):
                    df_train[col] = [
                        (
                            float(val)
                            if self._is_float(val)
                            else median if "sna" in val.lower() else 0
                        )
                        for val in df_train[col]
                    ]
                    df_train[col].fillna(median, inplace=True)

                    # df[col] = [
                    #     (
                    #         float(val)
                    #         if self._is_float(val)
                    #         else median if "sna" in val.lower() else 0
                    #     )
                    #     for val in df[col]
                    # ]
                    # df[col].fillna(median, inplace=True)
                else:
                    # df[col] = df[col].astype(str).fillna("nan")
                    df_train[col] = df_train[col].astype(str).fillna("nan")
            else:
                df_train[col] = [
                    "sna" if "sna" in val.lower() else str(val)
                    for val in df_train[col].astype(str)
                ]
                # df[col] = [
                #     "sna" if "sna" in val.lower() else str(val)
                #     for val in df[col].astype(str)
                # ]
                df_train[col] = df_train[col].astype(str).fillna("nan")
                # df[col] = df[col].astype(str).fillna("nan")

        return df_train

    def fill_sna_mean_rl(
        self, df_train: pd.DataFrame, th: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fills missing values in the DataFrame using the mean of columns where applicable, based on a threshold for unique values.

        Parameters:
        df (pd.DataFrame): The DataFrame to be cleaned.
        df_train (pd.DataFrame): Training data DataFrame used as a reference to calculate the mean.
        th (int): Threshold number of unique values required to consider a column for mean filling.

        Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the updated main DataFrame and training DataFrame.
        """
        for col in df_train.columns:
            if len(df_train[col].value_counts()) >= th:
                col_numeric = pd.to_numeric(df_train[col], errors="coerce")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    mean = col_numeric.mean(skipna=True)

                if self._is_float(mean) and not pd.isnull(mean):
                    df_train[col] = [
                        (
                            float(val)
                            if self._is_float(val)
                            else mean if "sna" in val.lower() else 0
                        )
                        for val in df_train[col]
                    ]
                    df_train[col].fillna(mean, inplace=True)

                    # df[col] = [
                    #     (
                    #         float(val)
                    #         if self._is_float(val)
                    #         else mean if "sna" in val.lower() else 0
                    #     )
                    #     for val in df[col]
                    # ]
                    # df[col].fillna(mean, inplace=True)
                else:
                    # df[col] = df[col].astype(str).fillna("nan")
                    df_train[col] = df_train[col].astype(str).fillna("nan")
            else:
                df_train[col] = [
                    "sna" if "sna" in val.lower() else str(val)
                    for val in df_train[col].astype(str)
                ]
                # df[col] = [
                #     "sna" if "sna" in val.lower() else str(val)
                #     for val in df[col].astype(str)
                # ]
                df_train[col] = df_train[col].astype(str).fillna("nan")
                # df[col] = df[col].astype(str).fillna("nan")

        return df_train

    def fill_sna_zero_rl(
        self, df_train: pd.DataFrame, th: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fills missing values in the DataFrame with zeros for columns based on a threshold for unique values.

        Parameters:
        df (pd.DataFrame): The DataFrame to be cleaned.
        df_train (pd.DataFrame): Training data DataFrame used as a reference.
        th (int): Threshold number of unique values required to consider a column for zero filling.

        Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the updated main DataFrame and training DataFrame.
        """
        for col in df_train.columns:
            if len(df_train[col].value_counts()) >= th:
                col_numeric = pd.to_numeric(df_train[col], errors="coerce")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    mean = col_numeric.mean(skipna=True)

                if self._is_float(mean) and not pd.isnull(mean):
                    df_train[col] = [
                        (
                            float(val)
                            if self._is_float(val)
                            else 0 if "sna" in val.lower() else 0
                        )
                        for val in df_train[col]
                    ]
                    df_train[col].fillna(0, inplace=True)

                    # df[col] = [
                    #     (
                    #         float(val)
                    #         if self._is_float(val)
                    #         else 0 if "sna" in val.lower() else 0
                    #     )
                    #     for val in df[col]
                    # ]
                    # df[col].fillna(0, inplace=True)
                else:
                    # df[col] = df[col].astype(str).fillna("nan")
                    df_train[col] = df_train[col].astype(str).fillna("nan")
            else:
                df_train[col] = [
                    "sna" if "sna" in val.lower() else str(val)
                    for val in df_train[col].astype(str)
                ]
                # df[col] = [
                #     "sna" if "sna" in val.lower() else str(val)
                #     for val in df[col].astype(str)
                # ]
                df_train[col] = df_train[col].astype(str).fillna("nan")
                # df[col] = df[col].astype(str).fillna("nan")

        return df_train

    def fill_sns_transform_mp(
        self, df: pd.DataFrame, sns_dict: dict = None
    ) -> pd.DataFrame:
        """
        Applies transformations to the DataFrame in a multiprocessing environment, optionally using a dictionary of settings.

        Parameters:
        df (pd.DataFrame): The DataFrame to be transformed.
        sns_dict (dict, optional): A dictionary containing settings for the transformations.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        split_dfs = split_df(df)

        # Apply the sns transform function to each split dataframe
        pool = mp.Pool()
        results = []
        for df in split_dfs:
            result = pool.apply_async(
                self.fill_sns_transform,
                args=(
                    df,
                    sns_dict,
                ),
            )
            results.append(result)

        # Merge the split dataframes back together
        processed_cols = [result.get() for result in results]

        processed_cols_df = [
            result[0] for result in processed_cols if not result[0].empty
        ]

        df = combine_dfs(processed_cols_df)

        pool.close()
        pool.join()

        return df
