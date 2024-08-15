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

from pandas import read_csv
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from numpy import percentile
import multiprocessing as mp
from data.utils import split_df, combine_dfs


class OutlierDetection:
    """
    Implements various methods for detecting and handling outliers within dataframes using statistical techniques
    and machine learning models. This class includes methods using Isolation Forest, Local Outlier Factor (LOF),
    interquartile range (IQR), and standard deviation.
    """

    def _isolation_forest(self, df_col: pd.Series) -> list[int]:
        """
        Applies the Isolation Forest algorithm to detect outliers in a given DataFrame column.

        Parameters:
        df_col (pd.Series): The DataFrame column to analyze.

        Returns:
        list[int]: List of outlier indices.
        """
        # df to array
        feature_data = df_col.values
        # identify outliers in feature data
        iso = IsolationForest(contamination=0.00005)
        yhat = iso.fit_predict(feature_data)
        # find index of outliers
        outliers = [i for i, e in enumerate(yhat) if e == -1]

        return outliers

    def _lof(self, df_col: pd.Series) -> list[int]:
        """
        Applies the Local Outlier Factor algorithm to detect outliers in a single column.

        Parameters:
        df_col (pd.Series): The DataFrame column to analyze.

        Returns:
        list[int]: List of outlier indices.
        """
        # df to array
        feature_data = df_col.values
        # identify outliers in feature data
        clf = LocalOutlierFactor(n_neighbors=2)
        yhat = clf.fit_predict(feature_data)
        # find index of outliers
        outliers = [i for i, e in enumerate(yhat) if e == -1]

        return outliers

    def _iqr(self, df_col: pd.Series) -> list[int]:
        """
        Uses the Interquartile Range (IQR) to detect outliers in a single column.

        Parameters:
        df_col (pd.Series): The DataFrame column to analyze.

        Returns:
        list[int]: List of outlier indices.
        """
        # df to array
        feature_data = df_col.values
        # calculate interquartile range
        q25, q75 = percentile(feature_data, 25), percentile(feature_data, 75)
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 5
        lower, upper = q25 - cut_off, q75 + cut_off

        # identify outliers
        outliers = [i for i, e in enumerate(feature_data) if e < lower or e > upper]
        # f = open("output.txt", "a")
        # print("----------")
        # print("Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f" % (q25, q75, iqr))
        # print("Identified outliers: %d" % len(outliers))
        # print("lower cut-off: ", lower, " upper cut-off: ", upper)
        # print("min: ", feature_data.min(), " max: ", feature_data.max())
        # f.close()

        return outliers

    def _sd(self, df_col: pd.Series) -> list[int]:
        """
        Detects outliers based on the standard deviation(sd) method in a single column.

        Parameters:
        df_col (pd.Series): The DataFrame column to analyze.

        Returns:
        list[int]: List of outlier indices.
        """
        # df to array
        feature_data = df_col.values.astype(float)

        std = feature_data.std()
        mean = feature_data.mean()

        # identify outliers
        cut_off = std * 3
        lower, upper = mean - cut_off, mean + cut_off

        # identify outliers
        outliers = [i for i, e in enumerate(feature_data) if e < lower or e > upper]
        # f = open("output.txt", "a")
        # print("----------")
        # print("Identified outliers: %d" % len(outliers))
        # print("lower cut-off: ", lower, " upper cut-off: ", upper)
        # print("min: ", feature_data.min(), " max: ", feature_data.max())
        # f.close()

        return outliers

    def outlier_detection(self, df, df_train) -> pd.DataFrame:
        """
        Processes entire DataFrame to detect and handle outliers using combined methods.

        Parameters:
        df (pd.DataFrame): DataFrame with potential outliers.
        df_train (pd.DataFrame): Training data, possibly containing outliers.

        Returns:
        pd.DataFrame: Processed DataFrame with outliers handled.
        """
        for i, col in enumerate(df_train.columns):
            if df_train[col].dtype != np.float32 and df_train[col].dtype != np.float64:
                continue
            else:
                outliers_iforest = self._isolation_forest(df_train[[col]])
                outliers_lof = self._lof(df_train[[col]])
                outliers_iqr = self._iqr(df_train[[col]])
                # outliers_sd = self._sd(df_train[[col]])

                set_outlier = list(
                    set(outliers_iforest) & set(outliers_lof) & set(outliers_iqr)
                )

                if len(set_outlier) > 0:
                    print(f"Column {i} out of {len(df.columns)}: {col}")
                    print("Num outliers: ", len(set_outlier))
                    df.loc[df[col].isin(set_outlier), col] = np.nan
                    df_train.loc[df_train[col].isin(set_outlier), col] = np.nan

                    df[col] = (
                        df[col].interpolate(method="linear", axis=0).ffill().bfill()
                    )
                    df_train[col] = (
                        df_train[col]
                        .interpolate(method="linear", axis=0)
                        .ffill()
                        .bfill()
                    )

                    """

                    # df = self.moving_avg(df, col, outliers_sd)
                    # df_train = self.moving_avg(df_train, col, outliers_sd)
                    # print("moving avg applied")

                    df = self.moving_avg(df, col, set_outlier)

                    df_train = self.moving_avg(
                        df_train,
                        col,
                        set_outlier,
                    )

                    # df_train[columns_toexcel].to_excel("./after_moving_avg.xlsx")

                    """

        return df, df_train

    def outlier_detection_mp(
        self, df: pd.DataFrame, df_train: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_dfs = split_df(df)
        split_df_trains = split_df(df_train)

        # Apply the decode function to each split dataframe
        pool = mp.Pool()
        results = []
        for df, df_train in zip(split_dfs, split_df_trains):
            result = pool.apply_async(
                self.outlier_detection,
                args=(
                    df,
                    df_train,
                ),
            )
            results.append(result)

        # Merge the split dataframes back together
        processed_cols = [result.get() for result in results]

        processed_cols_df = [
            result[0] for result in processed_cols if not result[0].empty
        ]
        processed_cols_df_train = [
            result[1] for result in processed_cols if not result[1].empty
        ]

        df_train = combine_dfs(processed_cols_df_train)
        df = combine_dfs(processed_cols_df)
        # Close the pool to prevent any more tasks from being submitted to the workers
        pool.close()
        pool.join()

        return df, df_train

    def _moving_avg(
        self, df: pd.DataFrame, col: pd.Series, row_ids: list[int]
    ) -> pd.DataFrame:
        # specify the column and rows you want to replace with moving average
        rows_to_replace = row_ids

        # calculate the moving average using rolling() function and store in a new column
        df["B_MA"] = df[col].rolling(window=3, center=True).mean()

        # use loc function to replace only the specified rows in the original column
        df.loc[rows_to_replace, col] = df["B_MA"][rows_to_replace]

        # drop the temporary 'B_MA' column
        df = df.drop("B_MA", axis=1)

        return df

    def _interpolate_nearest(self, df: pd.DataFrame) -> pd.DataFrame:
        df.interpolate(method="linear", axis=0).ffill().bfill()
        return df

    def _interpolate_spline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(
            df.interpolate(method="spline", order=1, limit=None, limit_direction="both")
        )

        return df
