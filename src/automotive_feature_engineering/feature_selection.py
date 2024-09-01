# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
import re
from pathlib import Path
import pathlib
from typing import List, Tuple
import eli5
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import automotive_feature_engineering.utils.utils as utils

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
from automotive_feature_engineering.utils.utils import split_df, combine_dfs


class FeatureSelection:
    def __init__(self):
        pass

    ##########################################
    # Drop highly correlated signals from df_train and final df
    ##########################################
    def _get_corr_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the correlation matrix for the given DataFrame.

        Parameters:
        df (pd.DataFrame): Dataframe from which to compute the correlation matrix.

        Returns:
        pd.DataFrame: Correlation matrix of the dataframe.
        """
        return pd.DataFrame(
            np.abs(np.corrcoef(df.values, rowvar=False)),
            columns=df.columns,
            index=df.columns,
        )

    def _get_target_corrs(
        self, features: pd.DataFrame, targets: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate correlations between features and targets.

        Parameters:
        features (pd.DataFrame): Features dataframe.
        targets (pd.DataFrame): Targets dataframe.

        Returns:
        pd.DataFrame: Correlation matrix between features and targets.
        """
        target_names = targets.columns
        combined_df = combine_dfs([features, targets])
        corr_matrix = self._get_corr_matrix(combined_df)
        return corr_matrix[target_names].drop(labels=target_names)

    def _get_sorted_target_corrs(
        self, features: pd.DataFrame, targets: pd.DataFrame, sort_by_max: bool
    ) -> pd.Series:
        """
        Sort features by their correlation with the target, either by max or min correlation.

        Parameters:
        features (pd.DataFrame): Features dataframe.
        targets (pd.DataFrame): Targets dataframe.
        sort_by_max (bool): If True, sort by maximum correlation; otherwise, sort by minimum.

        Returns:
        pd.Series: Sorted series of correlations.
        """
        target_corrs = self._get_target_corrs(features, targets)

        if sort_by_max:
            return target_corrs.max(axis=1).sort_values(ascending=False)

        return target_corrs.min(axis=1).sort_values(ascending=False)

    def _sort_features_by_target_corrs(
        self, features: pd.DataFrame, targets: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Sorts the features DataFrame based on the correlation of its columns with the target DataFrame's columns, in descending order.

        Parameters:
        features (pd.DataFrame): The DataFrame containing the feature variables.
        targets (pd.DataFrame): The DataFrame containing the target variables.

        Returns:
        pd.DataFrame: A DataFrame of features sorted by their correlation strength with the target variables, with the most correlated features appearing first.
        """
        sorted_target_corrs = self._get_sorted_target_corrs(
            features, targets, sort_by_max=True
        )
        return features[sorted_target_corrs.index]

    def _get_noncorrelating_features(
        self,
        sorted_features: pd.DataFrame,
        threshold: float,
    ) -> List[str]:
        feature_corrs = self._get_corr_matrix(sorted_features)
        mask = np.triu(feature_corrs)
        np.fill_diagonal(mask, False)

        selected_features = feature_corrs[mask.max(axis=0) <= threshold].index
        return list(selected_features)

    ##########################################
    # drop correlated features
    ##########################################
    def drop_correlated_features_fit(
        self,
        df_features: pd.DataFrame,
        th: float,
        importances,
    ) -> list[str]:
        """
        Identifies and lists features to drop based on a given correlation threshold and their relative importances.

        Parameters:
        df_features (pd.DataFrame): DataFrame containing the features to analyze.
        th (float): Correlation threshold; pairs of features with a correlation above this value will be considered for removal.
        importances (list of tuples): List of tuples where each tuple contains the importance score and the feature name.

        Returns:
        list[str]: List of feature names recommended to be dropped due to high correlation.
        """

        # Calculate the correlation matrix
        corr_matrix = df_features.corr()

        # Identify highly correlated features
        highly_correlated_features = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= th:
                    highly_correlated_features.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j])
                    )

        # Drop less important features from each highly correlated pair
        feature_importance_dict = {
            feature: importance for importance, feature in importances
        }
        drop_cols_corr = []  # Track the removed features

        for feature1, feature2 in highly_correlated_features:
            if feature1 in drop_cols_corr or feature2 in drop_cols_corr:
                continue  # Skip if any feature has already been removed

            if feature_importance_dict[feature1] < feature_importance_dict[feature2]:
                # df_features = df_features.drop(feature1, axis=1)
                drop_cols_corr.append(feature1)
            else:
                # df_features = df_features(feature2, axis=1)
                drop_cols_corr.append(feature2)
        print(f"Features to be dropped due to high correlation: {drop_cols_corr}")

        return drop_cols_corr

    def drop_correlated_features_transform(
        self,
        df: pd.DataFrame,
        drop_cols_corr: list,
    ) -> pd.DataFrame:
        """
        Applies the feature removal identified by `drop_correlated_features_fit` to a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame from which to remove the features.
        drop_cols_corr (list): List of column names to be dropped.

        Returns:
        pd.DataFrame: The DataFrame after removing the specified columns.
        """
        df = df.drop(columns=drop_cols_corr, axis=1, errors="ignore")

        return df

    ##########################################
    # remove columns with only one distinct value
    ##########################################
    def _filter_unique_values_fit(self, df: pd.DataFrame) -> list[str]:
        """
        Identifies columns with only one unique value in the DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame to analyze.

        Returns:
        list[str]: List of columns with only one unique value.
        """
        drop_col_var = [col for col in df.columns if len(df[col].unique()) <= 1]

        return drop_col_var

    def _filter_unique_values_transform(
        self, df: pd.DataFrame, drop_col_var: list[str] = None
    ) -> pd.DataFrame:
        """
        Removes specified columns from a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which columns will be removed.
        - drop_col_var (list[str], optional): A list of strings representing the column names to be dropped. Defaults to None, meaning no columns will be dropped if not specified.

        Returns:
        - pd.DataFrame: The modified DataFrame with the specified columns removed.
        """
        df = df.drop(columns=drop_col_var, axis=1, errors="ignore")
        return df

    ##########################################
    # remove columns below variance threshold
    ##########################################

    def filter_by_variance(
        self, df: pd.DataFrame, df_train: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters the given DataFrames by removing columns with low variance in the training set.

        Parameters:
        - df (pd.DataFrame): DataFrame to be filtered based on the training set variance.
        - df_train (pd.DataFrame): Training DataFrame used to compute variance thresholds.

        Returns:
        - tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the filtered DataFrame and the filtered training DataFrame.
        """
        # keep target features
        has_high_variance = df_train.var(numeric_only=True) > 0.1

        df_train = df_train.loc[
            :, has_high_variance.reindex(df_train.columns, axis=1, fill_value=True)
        ]

        df = df.loc[:, has_high_variance.reindex(df.columns, axis=1, fill_value=True)]

        return df, df_train

    def filter_by_variance_fit(self, df: pd.DataFrame, th=0.1) -> list[str]:
        """
        Determines which columns to drop based on the variance threshold.

        Calculates the variance for numeric columns in the DataFrame and identifies those that are below
        a specified threshold (`th`). It ensures at least 5 columns are retained regardless of their variance.

        Parameters:
        - df (pd.DataFrame): DataFrame from which to calculate variance.
        - th (float): Variance threshold; columns with variance below this value are candidate for dropping.

        Returns:
        - list[str]: List of column names that should be dropped.
        """
        drop_col = (
            df.select_dtypes(include=["number"])
            .columns[df.var(numeric_only=True) <= th]
            .tolist()
        )

        # Calculate variances for numeric columns
        variances = df.select_dtypes(include=["number"]).var()

        # Sort variances in descending order
        variances_sorted = variances.sort_values(ascending=False)

        # Determine how many columns to keep
        num_columns_to_keep = max(len(variances_sorted) - len(drop_col), 5)

        # Select columns to drop and ensure keeping at least 5 columns if needed
        if num_columns_to_keep < 5:
            drop_col_var = (
                drop_col + variances_sorted.index[: 5 - num_columns_to_keep].tolist()
            )
        else:
            drop_col_var = drop_col

        return drop_col_var

    def filter_by_variance_transform(
        self, df: pd.DataFrame, drop_col_var=None
    ) -> pd.DataFrame:
        """
        Removes specified columns from a DataFrame based on the variance filter.

        Parameters:
        - df (pd.DataFrame): DataFrame from which columns will be dropped.
        - drop_col_var (list[str], optional): List of column names to be dropped; defaults to None.

        Returns:
        - pd.DataFrame: The DataFrame after dropping the specified columns.
        """
        df = df.drop(columns=drop_col_var, axis=1, errors="ignore")
        return df

    ##########################################
    # Drop Signals containing a substring
    ##########################################
    def _dropSignal(self, df, substring):
        """
        Drops columns from the DataFrame that contain the specified substring in their names.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which columns will be dropped.
        - substring (str): Substring to look for in the column names.

        Returns:
        - pd.DataFrame: The modified DataFrame with the specified columns removed.
        """
        # print("Number of features in data set:",len(df.columns))
        print(f"Dropping all {substring} signals...")
        counter = 0
        for colname in df.columns:
            if re.search(substring, colname, re.IGNORECASE):
                df.drop(str(colname), axis=1, inplace=True)
                counter += 1
            # if str(colname).find(substring) != -1:
            #    df.drop(str(colname), axis=1, inplace = True)
            #    counter += 1
            else:
                pass
        print(f"Found {counter} Signals containing {substring}")
        print(f"All {substring} signals dropped!")
        print("Number of features in data set:", len(df.columns))
        return df

    def elimSignals(self, df, blacklist):
        """
        Eliminates columns from the DataFrame that match any item in a given blacklist.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which columns will be eliminated.
        - blacklist (list): A list of substrings used to identify columns to be removed.

        Returns:
        - pd.DataFrame: The DataFrame after all specified columns have been removed.
        """
        print("Eliminating Signals from a Blacklist...")

        for item in blacklist:
            df = self._dropSignal(df, item)
            # print("Number of features in data set:",len(df.columns))
        return df

    def elim_signals(self, df, blacklist) -> pd.DataFrame:
        """
        Removes columns from the DataFrame based on a blacklist of regex patterns.

        Unlike `elimSignals`, this method utilizes Pandas built-in `filter` with regex to identify and drop
        columns directly, which can be more efficient for larger DataFrames with complex blacklist patterns.

        Parameters:
        - df (pd.DataFrame): DataFrame from which to remove columns.
        - blacklist (str): Regex pattern representing the blacklist of column names.

        Returns:
        - pd.DataFrame: The DataFrame after the specified columns have been removed.
        """
        df = df[df.columns.drop(list(df.filter(regex=re.compile(blacklist))))]

        return df

    ##########################################
    # Global Feature Importance Calculation
    ##########################################
    def calc_globalFeatureImportance(
        self, path, model, feature_df, target_df, alt_config=None
    ):
        """
        Calculates global feature importances using specified tree-based models and configurations.
        This method supports Extra Trees and Random Forest regressors.

        Parameters:
        - path (str): Path where feature importance results will be saved.
        - model (str): Model type ('etree' or 'randomforest').
        - feature_df (pd.DataFrame): DataFrame containing feature data.
        - target_df (pd.DataFrame): DataFrame containing target data.
        - alt_config (dict, optional): Alternative configuration for the model.

        Returns:
        - dict: A dictionary where keys are feature importances and values are corresponding feature names.
        """
        print("---Calculating global Feature Importances---")
        result_dict = {}
        if model == "etree":
            config_dict = utils.get_dict(
                config_path=self.alt_config,
                default_path=pathlib.Path(__file__).resolve().parent.parent.as_posix()
                + "/src/automotive_feature_engineering/reinforcement_learning/rl_etree_defaults.json",
            )
            regr = ExtraTreesRegressor(**config_dict)
            regr.fit(feature_df, target_df)
            print("---Global Feature Importance calculated for ETreesRegressor---")
            importances = regr.feature_importances_
            # for k,v in sorted(zip(importances, preprocessor.getFeatureColumns()), reverse=True):
            #     if k > 0.000999:
            #         print(f"{k:.3f} {v:20}\n")
            result_dict = sorted(zip(importances, feature_df.columns), reverse=True)

            # Write Global Feature Importances to File:
            with open(
                os.path.join(
                    path, f"{target_df.columns}-Global_FeatureImportances.txt"
                ),
                "w",
            ) as f:
                for k, v in sorted(zip(importances, feature_df.columns), reverse=True):
                    f.write(f"{k:.3f} {v:20}\n")
        elif model == "randomforest":
            config_dict = utils.get_dict(
                config_path=alt_config,
                default_path=(
                    pathlib.Path(__file__).resolve().parent.parent.as_posix()
                    + "/automotive_feature_engineering/reinforcement_learning/rl_randomforest_defaults.json"
                ),
            )
            print(config_dict)
            regr = RandomForestRegressor(**config_dict)
            regr.fit(feature_df, target_df)
            print(
                "---Global Feature Importance calculated for RandomForestRegressor---"
            )
            importances = regr.feature_importances_
            # Print Importances > 0.000999
            # for k, v in sorted(
            #    zip(importances, preprocessor.getFeatureColumns()), reverse=True
            # ):
            #    if k > 0.000999:
            #        print(f"{k:.3f} {v:20}\n")
            result_dict = sorted(zip(importances, feature_df.columns), reverse=True)

            # Write Global Feature Importances to File:
            with open(
                os.path.join(
                    path, f"{target_df.columns.values}-Global_FeatureImportances.txt"
                ),
                "w",
            ) as f:
                for k, v in sorted(zip(importances, feature_df), reverse=True):
                    f.write(f"{k:.3f} {v:20}\n")
        else:
            print("Regressor not suitable for Feature Importance calculation")
        print("---Global Feature Importance calculated---")
        print("Features and their corresponding importances: ")
        for k, v in result_dict:
            if k > 0.000999:
                print(f"{k:.3f} {v:20}\n")

        return result_dict

    ##########################################
    # Drop Signals from a pre-defined importance dictionary
    ##########################################
    def dropImportances(self, df, importances, number):
        """
        Drops all columns except the top 'number' most important according to a provided importance list.

        Parameters:
        - df (pd.DataFrame): DataFrame from which to drop columns.
        - importances (list of tuples): List containing tuples of (importance, column name).
        - number (int): Number of most important signals to retain.

        Returns:
        - pd.DataFrame: The DataFrame after less important columns have been dropped.
        """
        print(f"Dropping all signals except the {number} most important.")
        # print("Number of features in data set:", len(df.columns))
        counter = len(importances)
        importances_list = list(importances)
        for i in range(len(importances_list), number, -1):
            colname = importances_list[i - 1][1]
            # print("Dropping: ", colname)
            df.drop(str(colname), axis=1, inplace=True)
            counter -= 1
        # print("Remaining signals: ")
        # print(list(df.columns))
        # print(f"Counter at: {counter}")
        print("Number of features in data set:", len(df.columns))
        return df

    ##########################################
    # Drop Signals from a pre-defined importance dictionary that are 0
    ##########################################
    def dropUnimportantFeatures(self, df, importances, importance):
        """
        Drops all features with an importance below a specified threshold from the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame from which to drop columns.
        - importances (list of tuples): List containing tuples of (importance, column name).
        - importance (float): Threshold below which features are considered unimportant.

        Returns:
        - pd.DataFrame: The DataFrame after unimportant features have been dropped.
        """
        print("Dropping all features with an importance of 0.")
        print("Number of features in data set:", len(df.columns))

        # create boolean mask to select columns to keep
        keep_cols = ~df.columns.isin([col for _, col in importances if _ < importance])
        drop_cols = df.columns[~keep_cols]
        # select columns to keep and assign back to dataframe
        df = df.loc[:, keep_cols]

        print("All unimportant features dropped.")
        print("Number of features in data set:", len(df.columns))
        return df

    ##########################################
    # Drop Signals from a pre-defined importance dictionary that are 0
    ##########################################
    def drop_unimportant_features_fit(
        self, docu_path, th, df_train_features, df_train_target, alt_config=None
    ):
        """
        Calculates feature importances and identifies columns to drop based on a threshold.

        Parameters:
        - docu_path (str): Path for documentation or output.
        - th (float): Threshold for dropping features. Features with importance below this threshold will be dropped.
        - df_train_features (pd.DataFrame): Features DataFrame used for training the model.
        - df_train_target (pd.DataFrame): Target DataFrame used for training the model.
        - alt_config (dict, optional): Alternate configuration for the RandomForest model.

        Returns:
        - list: List of column names that should be dropped based on the importance threshold.
        """
        feature_selection = FeatureSelection()
        # Determine global feature importance on train data:
        importances = feature_selection.calc_globalFeatureImportance(
            docu_path,
            "randomforest",
            df_train_features,
            df_train_target,
            alt_config=alt_config,
        )
        # create boolean mask to select columns to keep
        keep_cols_fi = ~df_train_features.columns.isin(
            [col for _, col in importances if _ < th]
        )
        drop_cols_fi = df_train_features.columns[~keep_cols_fi]
        drop_cols_fi = [col for col in drop_cols_fi if col != ""]
        return drop_cols_fi

    ##########################################
    # Drop Signals from a pre-defined importance dictionary that are 0
    ##########################################
    def drop_unimportant_features_transform(self, df, drop_cols_fi=None):
        """
        Drops specified columns from a DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame from which to drop columns.
        - drop_cols_fi (list, optional): List of column names to be dropped.

        Returns:
        - pd.DataFrame: Modified DataFrame with specified columns removed.
        """
        print("Number of features in data set:", len(df.columns))
        # select columns to keep and assign back to dataframe
        df = df.drop(columns=drop_cols_fi, axis=1, errors="ignore")
        print("All unimportant features dropped.")
        print("Number of features in data set:", len(df.columns))
        if len(df.columns) == 0:
            raise ValueError("#ERROR#! No important features could be found!")
        return df

    ##########################################
    # Pearson Correlation
    ##########################################
    def pearson(self, df, path, make_heatmap):
        """
        Calculates the Pearson correlation coefficients and optionally creates a heatmap.

        Parameters:
        - df (pd.DataFrame): DataFrame for which correlations are calculated.
        - path (str): Path where to save the correlation heatmap.
        - make_heatmap (bool): Whether to create and save a heatmap of the correlations.

        Returns:
        - pd.DataFrame: DataFrame of correlation coefficients.
        """
        if make_heatmap:
            pearson_correl = df.corr()
            plt.figure(figsize=(20, 16))
            heatmap = sns.heatmap(pearson_correl, vmin=-1, vmax=1, annot=True)
            heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12)
            plt.savefig(os.path.join(path, "PearsonCorrel.png"))
            plt.clf()
        else:
            pearson_correl = df.corr()
        return pearson_correl

    ##########################################
    # Pearson Correlation with targets
    ##########################################
    def get_feature_and_target_corrs(
        self,
        path: str,
        df: pd.DataFrame,
        target_names: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates and separates the correlations of features with each other and with targets.

        Parameters:
        - path (str): Path where outputs (if any) will be saved.
        - df (pd.DataFrame): DataFrame containing both feature and target data.
        - target_names (List[str]): List of column names considered as targets.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Tuple of DataFrames, the first being correlations among features, and the second being correlations between features and targets.
        """
        corr_matrix = self.pearson(df, path, False)
        corr_matrix_adjusted = corr_matrix.loc[~corr_matrix.index.isin(target_names)]
        feature_corrs = corr_matrix_adjusted.loc[
            :, ~corr_matrix_adjusted.columns.isin(target_names)
        ]
        target_corrs = corr_matrix_adjusted.loc[
            :, corr_matrix_adjusted.columns.isin(target_names)
        ]

        return feature_corrs, target_corrs

    def plot_target_corr_heatmap(
        self,
        target_names: List[str],
        path: str,
        data: pd.DataFrame,
    ):
        """
        Generates a correlation heatmap for both feature-feature and feature-target correlations.

        Parameters:
        - target_names (List[str]): List of names of the target variables.
        - path (str): Directory path where the heatmap will be saved.
        - data (pd.DataFrame): DataFrame containing both features and target variables.
        """
        feature_corrs, target_corrs = self.get_feature_and_target_corrs(
            path,
            data,
            target_names,
        )

        fig, (ax1, ax2) = plt.subplots(
            ncols=2,
            width_ratios=[feature_corrs.shape[1], target_corrs.shape[1]],
            figsize=(
                len(data.columns) / 10 if len(data.columns) > 60 else 20,
                len(data.columns) / 10 if len(data.columns) > 60 else 20,
            ),
            layout="constrained",
        )

        mask = np.triu(feature_corrs)
        np.fill_diagonal(mask, False)
        sns.heatmap(
            feature_corrs,
            annot=True,
            fmt=".3f",
            ax=ax1,
            cbar=False,
            cmap="Blues",
            mask=mask,
        )
        sns.heatmap(
            target_corrs, annot=True, fmt=".3f", ax=ax2, cbar=False, cmap="Blues"
        )

        plt.tick_params(left=False)
        ax1.set_title("Features")
        ax1.set_xticks(np.arange(len(feature_corrs.columns)))
        ax1.set_yticks(np.arange(len(feature_corrs.index)))
        ax1.set_xticklabels(feature_corrs.columns, rotation=90)
        ax1.set_yticklabels(feature_corrs.index, rotation=0)

        ax2.set_title("Targets")
        ax2.set(ylabel=None)
        ax2.set_xticks(np.arange(len(target_corrs.columns)))
        ax2.set_yticks(np.arange(len(target_corrs.index)))
        ax2.set_xticklabels(target_corrs.columns, rotation=90)
        ax2.set_yticklabels([], rotation=0)

        fig.savefig(os.path.join(path, "PearsonCorrel-with-Target.png"))

    ##########################################
    # Drop Signals containing a substring
    ##########################################
    def dropSignal(self, df, substring):
        """
        Drops columns from a DataFrame based on a substring match in column names.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which columns are to be dropped.
        - substring (str): The substring that if found in column names, will lead to the column being dropped.

        Returns:
        - pd.DataFrame: The modified DataFrame after dropping specified columns.
        """
        # print("Number of features in data set:",len(df.columns))
        print(f"Dropping all {substring} signals...")
        counter = 0
        for colname in df.columns:
            if re.search(substring, colname, re.IGNORECASE):
                df.drop(str(colname), axis=1, inplace=True)
                counter += 1
            # if str(colname).find(substring) != -1:
            #    df.drop(str(colname), axis=1, inplace = True)
            #    counter += 1
            else:
                pass
        print(f"Found {counter} Signals containing {substring}")
        print(f"All {substring} signals dropped!")
        print("Number of features in data set:", len(df.columns))
        return df

    def elimSignals(self, df, blacklist):
        """
        Eliminates columns from the DataFrame that contain any of the substrings specified in the blacklist.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which columns are to be eliminated.
        - blacklist (list of str): A list of substrings that if found in column names, will lead to the column being dropped.

        Returns:
        - pd.DataFrame: The DataFrame after all specified columns have been removed.
        """
        print("Eliminating Signals from a Blacklist...")
        for item in blacklist:
            df = self.dropSignal(df, item)
            # print("Number of features in data set:",len(df.columns))
        return df

    ##############################################
    # This is also a mean of explanation (XAI)
    ##############################################
    ##########################################
    # Calculate Permutation Importance - Local Explanation Method
    ##########################################
    def permImportance(
        self, model, path, df_train_features, df_train_target, index, alt_config=None
    ):
        """
        Calculates and visualizes permutation importance for features using the specified model.

        Parameters:
        - model (str): Model type ('etree' or 'randomforest').
        - path (str): Path where the results will be saved.
        - df_train_features (pd.DataFrame): Features DataFrame used for the model training.
        - df_train_target (pd.DataFrame): Target DataFrame used for the model training.
        - index (int): Index for a specific data point for local explanation.
        - alt_config (dict, optional): Alternate configuration for the model.

        Uses ELI5 library for visualization and explanation of feature importance and prediction.
        """
        print("---Calc ELI5 Permutation Importance---")
        perm = None
        if model == "etree":
            config_dict = utils.get_dict(
                config_path=alt_config,
                default_path=pathlib.Path(__file__).parent.parent.resolve().as_posix()
                + "/default_configs/etree_defaults.json",
            )
            X = df_train_features
            y = df_train_target
            regr = ExtraTreesRegressor(**config_dict).fit(X, y)
            perm = PermutationImportance(
                regr, random_state=config_dict.get("random_state")
            ).fit(X, y)
            print("---Permutation Importance fitted!---")
            # print(preprocessor.getFeatureColumns())
            ex_weights = eli5.explain_weights(
                perm, feature_names=df_train_features.columns.tolist()
            )
            # print(eli5.format_as_text(ex_weights))
            # print(ex_weights)
            html = eli5.format_as_html(ex_weights)
            self.writeToHtml(html, "ELI5_ExplainWeights", path)

            # Local Prediction explanation using ELI5:
            ex_red = eli5.explain_prediction(
                regr, X.iloc[index], feature_names=df_train_features.columns.tolist()
            )
            html = eli5.format_as_html(ex_red)
            self.writeToHtml(html, "ELI5_Explain_Pred", path)

        elif model == "randomforest":
            config_dict = utils.get_dict(
                config_path=alt_config,
                default_path=pathlib.Path(__file__).parent.parent.resolve().as_posix()
                + "/automotive_feature_engineering/reinforcement_learning/rl_randomforest_defaults.json",
            )
            X = df_train_features
            y = df_train_target

            regr = RandomForestRegressor(**config_dict).fit(X, y)
            perm = PermutationImportance(
                regr, random_state=config_dict.get("random_state")
            ).fit(X, y)
            print("---Permutation Importance fitted!---")
            # print(preprocessor.getFeatureColumns())
            ex_weights = eli5.explain_weights(
                perm, feature_names=df_train_features.columns.tolist()
            )
            # print(eli5.format_as_text(ex_weights))
            print(ex_weights)
            html = eli5.format_as_html(ex_weights)
            self.writeToHtml(html, "ELI5_ExplainWeights", path)

            # Local Prediction explanation using ELI5:
            ex_red = eli5.explain_prediction(
                regr, X.iloc[index], feature_names=df_train_features.columns.tolist()
            )
            html = eli5.format_as_html(ex_red)
            self.writeToHtml(html, "ELI5_Explain_Pred", path)

        else:
            print("Modeltype not yet supported for Permuation Importance.")
        return None

    def writeToHtml(self, obj, name, path):
        """
        Writes an object to an HTML file.

        Parameters:
        - obj (str): HTML content to be saved.
        - name (str): Name of the file.
        - path (str): Path where the file will be saved.
        """
        with open(os.path.join(path, f"{name}.html"), "w") as f:
            for k in obj:
                f.write(k)
