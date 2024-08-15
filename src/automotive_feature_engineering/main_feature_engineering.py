# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
import re
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.automotive_feature_engineering.feature_extraction import FeatureExtraction
from src.automotive_feature_engineering.feature_encoding import FeatureEncoding
from src.automotive_feature_engineering.feature_selection import FeatureSelection
from src.automotive_feature_engineering.feature_scaling import FeatureScaling
from src.automotive_feature_engineering.feature_interactions import FeatureInteractions
from src.automotive_feature_engineering.sna_handling import SnaHandling
from src.automotive_feature_engineering.utils import combine_dfs, get_feature_df
from joblib import dump, load

from timeit import default_timer as timer
import logging

# Setup the logger
# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class FeatureEngineering:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        model,
        target_names_list,
        import_joblib_path=None,
        alt_docu_path=None,
        alt_config=None,
        unrelated_cols=None,
        model_export=False,
        fe_export_joblib=False,
        explainable=False,
    ):
        """
        Initialize the Feature Engineering class.

        Parameters:
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        model: Model to be used for feature selection.
        target_names_list (List[str]): List of target names.
        import_joblib_path (str, optional): Path to import joblib file of previously exported feature engineering methods.
        alt_docu_path (str, optional): Alternative documentation path.
        alt_config (Dict, optional): Alternative configuration dictionary.
        unrelated_cols (List[str], optional): List of columns that are not part of the raw data.
        model_export (bool): Whether to export the model.
        fe_export_joblib (bool): Whether to export the feature engineering methods used.
        explainable (bool): Indicate if explainable AI shall be used.
        """

        self.df_train = df_train  # Training data
        self.model = model  # Model to be used for feature selection
        self.target_names_list = target_names_list  # List of target names
        self.alt_docu_path = alt_docu_path  # Alternative path to documentation otherwise the default path is used (where the script is located)
        self.export_list = []  # List to store the actions that were performed
        self.unrelated_cols = (
            unrelated_cols  # Columns that are not part of the raw data
        )
        self.df_test = df_test  # Test data
        self.model_export = model_export  # Export the model (boolean)
        self.fe_export_joblib = (
            fe_export_joblib  # Export the feature engineering methods used (boolean)
        )
        self.explainable = (
            explainable  # Indicate if explainable AI shall be used (boolean)
        )
        self.alt_config = alt_config  # Path to alternative configuration file (other than the default configuration file)
        self.import_joblib_path = import_joblib_path  # Path to import joblib file of previously exported feature engineering methods

        if alt_docu_path == None:
            self.docu_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.docu_path = alt_docu_path
            # check if the path exists, if not create it
            if not os.path.exists(self.docu_path):
                os.makedirs(self.docu_path)

        logger.info(f"Alt Doku Path: {self.docu_path}")
        logger.info(
            "Feature Engineering shall only be performed on the training data. The test data will be transformed accordingly afterwards."
        )
        logger.debug(
            f"Shape of training data before feature engineering: {self.df_train.shape} "
        )

        #######################
        ### Remove unrelated columns (e.g. timestamps) before processing
        #######################
        # This separates the target columns from the feature columns
        # self.df_test, self.df_test_target = get_feature_df(self.df_test, target_names_lists=self.target_names_list)

        self.df_train_working = self.df_train.copy()
        # remove timestamps
        self.df_train_working = self.clear_df_train(
            self.df_train_working, clear_cols=self.unrelated_cols
        )
        self.df_train_working_features, self.df_train_working_target = get_feature_df(
            self.df_train_working, target_names_list=self.target_names_list
        )

        logger.debug(
            f"Training data df_train: {self.df_train.shape}, df_train_working: {self.df_train_working.shape}"
        )
        logger.debug(
            f"df_train_working_features: {self.df_train_working_features.shape}, df_train_working_target: {self.df_train_working_target.shape}"
        )

    def main(self, method_list=None):
        """
        Main method to execute the feature engineering pipeline.

        Parameters:
        method_list (List[int], optional): List of method numbers to execute in the pipeline. If None, a default pipeline is used.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Transformed training and test dataframes.
        """
        # shuffle the columns
        # useful for multiprocessing that splits are randomized
        # df = df.sample(frac=1, axis=1)

        # Dictionary mapping numbers to functions
        function_dict = {
            0: (self._, ()),
            1: (
                self._drop_correlated_features_09,
                (),
            ),
            2: (
                self._drop_correlated_features_095,
                (),
            ),
            3: (self._sns_handling_median_8, ()),
            4: (self._sns_handling_median_32, ()),
            5: (self._sns_handling_mean_8, ()),
            6: (self._sns_handling_mean_32, ()),
            7: (self._sns_handling_zero_8, ()),
            8: (self._sns_handling_zero_32, ()),
            9: (self._filter_by_variance, ()),
            10: (self._ohe, ()),
            11: (self._feature_importance_filter_00009999, ()),
            12: (
                self._feature_importance_filter_00049999,
                (),
            ),
            13: (self._pca, ()),
            14: (self._polynominal_features, ()),
            # 15: (self._minmax_scaling, ()),
            # 16: (self._standard_scaling, ()),
            # 17: (self._robust_scaling, ()),
            ### always called at the beginning
            99: (self._filter_by_variance_0, ()),
        }

        ### Load actions from joblib file
        if self.import_joblib_path != None:
            import_list = load(self.import_joblib_path)

            for method in import_list:
                function, parameters = function_dict[method["action_num"]]
                function(*parameters, method["params"])
        else:
            ### Run actions
            if method_list == None:
                #######################
                ### Remove signals without variance since they are not adding any information to the model
                #######################
                self._filter_by_variance_0()
                # static pipelines according zu MuellerEtAl XXX paper
                # A pipeline without polynomial features needs to be used when the model is exported, explainable or the feature engineering is exported
                if self.explainable or self.model_export or self.fe_export_joblib:
                    pipeline = [4, 9, 10, 11]
                else:
                    pipeline = [4, 9, 10, 11, 14, 11]

                for number in pipeline:
                    if number in function_dict:
                        function, parameters = function_dict[number]
                        function(*parameters)
                    else:
                        logger.warning(f"No function found for number {number}")
                if self.fe_export_joblib:
                    file = os.path.join(
                        self.docu_path,
                        f"feature_engineering.joblib",
                    )
                    dump(self.export_list, file, compress=1)
                    logger.info("Successfully exported Feature Engineering")
            else:
                #######################
                ### Remove signals without variance
                #######################
                self._filter_by_variance_0()
                pipeline = method_list

                for number in pipeline:
                    if number in function_dict:
                        function, parameters = function_dict[number]
                        function(*parameters)
                    else:
                        logger.warning(f"No function found for number {number}")
                file = os.path.join(
                    self.docu_path,
                    f"feature_engineering.joblib",
                )
                dump(self.export_list, file, compress=1)
                logger.info("Successfully exported Feature Engineering")

        #######################
        ### Plot final correlation matrix after processing
        #######################

        feature_selection = FeatureSelection()
        # Calculate pearson correlation on remaining features:
        feature_selection.pearson(
            self.df_train_working_features, self.docu_path, make_heatmap=True
        )

        feature_selection.plot_target_corr_heatmap(
            self.target_names_list, self.docu_path, self.df_train
        )

        # Calculate permutation importance on remaining features:
        feature_selection.permImportance(
            self.model,
            self.docu_path,
            self.df_train_working_features,
            self.df_train_working_target,
            10,
            alt_config=self.alt_config,
        )

        return self.df_train, self.df_test

    ###############################
    # Method to clear df_train from artificially added 'file' and any other columns that are not part of the raw data
    ###############################
    def clear_df_train(self, df, clear_cols=None):
        """
        Method to clear df_train from artificially added 'file' and any other columns that are not part of the raw data.

        Parameters:
        df (pd.DataFrame): DataFrame to be cleared.
        clear_cols (List[str], optional): List of columns to be cleared.

        Returns:
        pd.DataFrame: Cleared DataFrame.
        """
        # Since df_train won't be used to create splits, but to be worked with as a pure df
        # just without the test set, we need to drop 'file' and 'timestamps' columns from it
        # These columns are not part of the raw (input) data to the model.
        if "file" in df:
            df.drop(columns=["file"], inplace=True)
        if "timestamps" in df:
            df.drop(columns=["timestamps"], inplace=True)
        if clear_cols:
            df.drop(columns=clear_cols, inplace=True)
        return df

    #######################
    ### Dummy function
    #######################
    def _(self):
        """
        Dummy function.
        """
        return None

    #######################
    # Drop highly correlated features
    #######################
    # alt_config is a dictionary that can be passed to the calc_globalFeatureImportance function to change the default settings defined in reinforcement_learning folder.
    def _drop_correlated_features_09(self, drop_cols_corr_09=None, alt_config=None):
        """
        Drop highly correlated features with a correlation threshold of 0.9.

        Parameters:
        drop_cols_corr_09 (List[str], optional): List of columns to drop.
        alt_config (Dict, optional): Alternative configuration dictionary.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        feature_selection = FeatureSelection()
        if drop_cols_corr_09 == None:

            importances = feature_selection.calc_globalFeatureImportance(
                self.docu_path,
                "randomforest",
                self.df_train_working_features,
                self.df_train_working_target,
                alt_config,
            )
            ### Remove "file" and "I_" before processing
            # Not needed anymore!
            # self.df, _ = get_feature_df(self.df, fuse_prefix="I_")
            # self.df_train, _ = get_feature_df(self.df_train, fuse_prefix="I_")

            drop_cols_corr_09 = feature_selection.drop_correlated_features_fit(
                self.df_train_working_features, 0.9, importances
            )
            self.export_list.append(dict(action_num=1, params=drop_cols_corr_09))

        # Drop the identified features to drop from the training data
        self.df_train = feature_selection.drop_correlated_features_transform(
            self.df_train, drop_cols_corr_09
        )
        # Drop the identified features to drop from the test data as well
        self.df_test = feature_selection.drop_correlated_features_transform(
            self.df_test, drop_cols_corr_09
        )
        # Drop the identified features to drop from the global data frame (includes training & test data)
        self.df_train_working_features = (
            feature_selection.drop_correlated_features_transform(
                self.df_train_working_features, drop_cols_corr_09
            )
        )

    #######################
    # Drop highly correlated features
    #######################
    # alt_config is a dictionary that can be passed to the calc_globalFeatureImportance function to change the default settings defined in reinforcement_learning folder.
    def _drop_correlated_features_095(self, drop_cols_corr_095=None, alt_config=None):
        """
        Drop highly correlated features with a correlation threshold of 0.95.

        Parameters:
        drop_cols_corr_09 (List[str], optional): List of columns to drop.
        alt_config (Dict, optional): Alternative configuration dictionary.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        feature_selection = FeatureSelection()
        if drop_cols_corr_095 == None:

            importances = feature_selection.calc_globalFeatureImportance(
                self.docu_path,
                "randomforest",
                self.df_train_working_features,
                self.df_train_working_target,
                alt_config,
            )

            drop_cols_corr_095 = feature_selection.drop_correlated_features_fit(
                self.df_train_working_features, 0.95, importances
            )
            self.export_list.append(dict(action_num=2, params=drop_cols_corr_095))

        # Drop the identified features to drop from the training data
        self.df_train = feature_selection.drop_correlated_features_transform(
            self.df_train, drop_cols_corr_095
        )
        # Drop the identified features to drop from the test data as well
        self.df_test = feature_selection.drop_correlated_features_transform(
            self.df_test, drop_cols_corr_095
        )
        # Drop the identified features to drop from the global data frame (includes training & test data)
        self.df_train_working_features = (
            feature_selection.drop_correlated_features_transform(
                self.df_train_working_features, drop_cols_corr_095
            )
        )

    #######################
    # Fix NaN values
    #######################
    def _sns_handling_median_8(self, sna_dict=None):
        """
        Fix NaN values by filling with median values for columns that have a sufficient number of unique values above a threshold 'th' (8 in this case).

        Parameters:
        sna_dict (dict, optional): Dictionary for storing NaN handling information. If not provided, it will be computed based on the training and test datasets.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        sna_handling = SnaHandling()
        start_main = timer()
        if sna_dict == None:
            sna_dict_train = sna_handling.fill_sna_median_fit(self.df_train, 8)
            sna_dict_test = sna_handling.fill_sna_median_fit(self.df_test, 8)
            sna_dict = sna_dict_train | sna_dict_test
            self.export_list.append(dict(action_num=3, params=sna_dict))

        self.df_train = sna_handling.fill_sna_median_transform(self.df_train, sna_dict)
        self.df_train_working_features = sna_handling.fill_sna_median_transform(
            self.df_train_working_features, sna_dict
        )
        self.df_test = sna_handling.fill_sna_median_transform(self.df_test, sna_dict)

        end_main = timer()
        logger.info("Total Time taken fill_nan: %s seconds", end_main - start_main)
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    # Fix NaN values
    #######################
    def _sns_handling_median_32(self, sna_dict=None):
        """
        Fix NaN values by filling with median values for columns that have a sufficient number of unique values above a threshold 'th' (32 in this case).

        Parameters:
        sna_dict (dict, optional): Dictionary for storing NaN handling information. If not provided, it will be computed based on the training and test datasets.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        sna_handling = SnaHandling()
        start_main = timer()
        if sna_dict == None:
            sna_dict_train = sna_handling.fill_sna_median_fit(self.df_train, 32)
            sna_dict_test = sna_handling.fill_sna_median_fit(self.df_test, 32)
            sna_dict = sna_dict_train | sna_dict_test
            self.export_list.append(dict(action_num=4, params=sna_dict))

        self.df_train = sna_handling.fill_sna_median_transform(self.df_train, sna_dict)
        self.df_train_working_features = sna_handling.fill_sna_median_transform(
            self.df_train_working_features, sna_dict
        )
        self.df_test = sna_handling.fill_sna_median_transform(self.df_test, sna_dict)

        end_main = timer()
        logger.info("Total Time taken fill_nan: %s seconds", end_main - start_main)
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    # Fix NaN values
    #######################
    def _sns_handling_mean_8(self, sna_dict=None):
        """
        Fix NaN values by filling with mean values for columns that have a sufficient number of unique values above a threshold 'th' (8 in this case).

        Parameters:
        sna_dict (dict, optional): Dictionary for storing NaN handling information. If not provided, it will be computed based on the training and test datasets.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        sna_handling = SnaHandling()
        start_main = timer()
        if sna_dict == None:
            sna_dict_train = sna_handling.fill_sna_mean_fit(self.df_train, 8)
            sna_dict_test = sna_handling.fill_sna_mean_fit(self.df_test, 8)
            sna_dict = sna_dict_train | sna_dict_test
            self.export_list.append(dict(action_num=5, params=sna_dict))

        self.df_train = sna_handling.fill_sna_mean_transform(self.df_train, sna_dict)
        self.df_train_working_features = sna_handling.fill_sna_mean_transform(
            self.df_train_working_features, sna_dict
        )
        self.df_test = sna_handling.fill_sna_mean_transform(self.df_test, sna_dict)

        end_main = timer()
        logger.info("Total Time taken fill_nan: %s seconds", end_main - start_main)
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    # Fix NaN values
    #######################
    def _sns_handling_mean_32(self, sna_dict=None):
        """
        Fix NaN values by filling with mean values for columns that have a sufficient number of unique values above a threshold 'th' (32 in this case).

        Parameters:
        sna_dict (dict, optional): Dictionary for storing NaN handling information. If not provided, it will be computed based on the training and test datasets.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        sna_handling = SnaHandling()
        start_main = timer()
        if sna_dict == None:
            sna_dict_train = sna_handling.fill_sna_mean_fit(self.df_train, 32)
            sna_dict_test = sna_handling.fill_sna_mean_fit(self.df_test, 32)
            sna_dict = sna_dict_train | sna_dict_test
            self.export_list.append(dict(action_num=6, params=sna_dict))

        self.df_train = sna_handling.fill_sna_mean_transform(self.df_train, sna_dict)
        self.df_train_working_features = sna_handling.fill_sna_mean_transform(
            self.df_train_working_features, sna_dict
        )
        self.df_test = sna_handling.fill_sna_mean_transform(self.df_test, sna_dict)

        end_main = timer()
        logger.info("Total Time taken fill_nan: %s seconds", end_main - start_main)
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    # Fix NaN values
    #######################
    def _sns_handling_zero_8(self, sna_dict=None):
        """
        Fix NaN values by filling with zero for columns that have a sufficient number of unique values above a threshold 'th' (8 in this case).

        Parameters:
        sna_dict (dict, optional): Dictionary for storing NaN handling information. If not provided, it will be computed based on the training and test datasets.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        sna_handling = SnaHandling()
        start_main = timer()
        if sna_dict == None:
            sna_dict_train = sna_handling.fill_sna_zero_fit(self.df_train, 8)
            sna_dict_test = sna_handling.fill_sna_zero_fit(self.df_test, 8)
            sna_dict = sna_dict_train | sna_dict_test
            self.export_list.append(dict(action_num=7, params=sna_dict))

        self.df_train = sna_handling.fill_sna_zero_transform(self.df_train, sna_dict)
        self.df_train_working_features = sna_handling.fill_sna_zero_transform(
            self.df_train_working_features, sna_dict
        )
        self.df_test = sna_handling.fill_sna_zero_transform(self.df_test, sna_dict)

        end_main = timer()
        logger.info("Total Time taken fill_nan: %s seconds", end_main - start_main)
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    # Fix NaN values
    #######################
    def _sns_handling_zero_32(self, sna_dict=None):
        """
        Fix NaN values by filling with zero for columns that have a sufficient number of unique values above a threshold 'th' (32 in this case).

        Parameters:
        sna_dict (dict, optional): Dictionary for storing NaN handling information. If not provided, it will be computed based on the training and test datasets.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated training, test, and working feature DataFrames.
        """
        sna_handling = SnaHandling()
        start_main = timer()
        if sna_dict == None:
            sna_dict_train = sna_handling.fill_sna_zero_fit(self.df_train, 32)
            sna_dict_test = sna_handling.fill_sna_zero_fit(self.df_test, 32)
            sna_dict = sna_dict_train | sna_dict_test
            self.export_list.append(dict(action_num=8, params=sna_dict))

        self.df_train = sna_handling.fill_sna_zero_transform(self.df_train, sna_dict)
        self.df_train_working_features = sna_handling.fill_sna_zero_transform(
            self.df_train_working_features, sna_dict
        )
        self.df_test = sna_handling.fill_sna_zero_transform(self.df_test, sna_dict)

        end_main = timer()
        logger.info("Total Time taken fill_nan: %s seconds", end_main - start_main)
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    # remove columns containing only one distinct value
    #######################
    def _filter_by_variance_0(self, drop_col_var_0=None):
        """
        Removes columns with only one unique value across the training and test datasets.

        Parameters:
        drop_col_var_0 (list, optional): List of column names to drop. If None, identifies columns with a single unique value.
        """
        start_main = timer()
        feature_selection = FeatureSelection()
        if drop_col_var_0 == None:
            drop_col_var_0 = feature_selection._filter_unique_values_fit(self.df_train)
            # Append this action to export list
            self.export_list.append(dict(action_num=99, params=drop_col_var_0))
        self.df_train = feature_selection._filter_unique_values_transform(
            self.df_train, drop_col_var=drop_col_var_0
        )
        self.df_test = feature_selection._filter_unique_values_transform(
            self.df_test, drop_col_var=drop_col_var_0
        )
        self.df_train_working_features = (
            feature_selection._filter_unique_values_transform(
                self.df_train_working_features, drop_col_var=drop_col_var_0
            )
        )
        end_main = timer()
        logger.info(
            "Total Time taken filter_unique_values: %s seconds", end_main - start_main
        )
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    # drop low variance signals
    #######################
    def _filter_by_variance(self, drop_col_var_01=None):
        """
        Removes columns with variance below a specified threshold 0.1 across the training and test datasets.

        Parameters:
        drop_col_var_01 (list, optional): List of column names to drop based on low variance. If None, it calculates which columns to drop.
        """
        start_main = timer()
        feature_selection = FeatureSelection()
        if drop_col_var_01 == None:
            drop_col_var_01 = feature_selection.filter_by_variance_fit(
                self.df_train_working_features, th=0.1
            )
            self.export_list.append(dict(action_num=9, params=drop_col_var_01))

        self.df_train = feature_selection.filter_by_variance_transform(
            self.df_train, drop_col_var_01
        )
        self.df_test = feature_selection.filter_by_variance_transform(
            self.df_test, drop_col_var_01
        )
        self.df_train_working_features = feature_selection.filter_by_variance_transform(
            self.df_train_working_features, drop_col_var_01
        )
        end_main = timer()
        logger.info(
            "Total Time taken filter_unique_values: %s seconds", end_main - start_main
        )
        logger.debug(
            f"fill_nan: df_train: {self.df_train.shape} df_test: {self.df_test.shape}"
        )

    #######################
    ### One-Hot-Encoding
    #######################
    def _ohe(self, ohe_regr=None):
        """
        Applies one-hot encoding to categorical variables in the training and test datasets.

        Parameters:
        ohe_regr (transformer object, optional): Pre-fitted one-hot encoder. If None, fits a new encoder.
        """
        logger.info("Starting one-hot encoding...")
        start_main = timer()
        feature_encoding = FeatureEncoding()

        # check if self.df_train_working_features has string columns, if not skip OHE
        string_columns = list(
            self.df_train_working_features.select_dtypes(
                include=["object", "category"]
            ).columns
        )

        if string_columns:
            if ohe_regr == None:
                ohe_regr = feature_encoding.one_hot_encoding_fit(
                    self.df_train_working_features
                )
                logger.info("Done fitting one-hot encoding")
                self.export_list.append(dict(action_num=10, params=ohe_regr))

            self.df_train = feature_encoding.one_hot_encoding_transform(
                self.df_train, ohe_regr=ohe_regr
            )
            self.df_test = feature_encoding.one_hot_encoding_transform(
                self.df_test, ohe_regr=ohe_regr
            )

            self.df_train_working_features = (
                feature_encoding.one_hot_encoding_transform(
                    self.df_train_working_features, ohe_regr=ohe_regr
                )
            )
            # self.df_train_working = self.df_train.copy()
            # self.df_train_working = self.clear_df_train(self.df_train_working, clear_cols=self.unrelated_cols)
            # self.df_train_working_features, self.df_train_working_target = get_feature_df(self.df_train_working, target_names_list=self.target_names_list)

            # print(f"Self.df_train_working_features: {self.df_train_working_features.shape}")
            # print(f"Self.df_train_working_target: {self.df_train_working_target.shape}")
            # print(f"Self.df_train_working: {self.df_train_working.shape}")
            # print(f"Self.df_train: {self.df_train.shape}")
            # print(f"Self.df_test: {self.df_test.shape}")

            logger.info("Debug: After transforming df with one-hot encoding")
            try:
                with open("/proc/self/status") as f:
                    memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]
                    logger.warning(f"Memory usage: {int(memusage.strip())} kB")
            except:
                pass

        else:
            logger.info(
                "No object or category columns to encode. Skipping one-hot encoding."
            )

        end_main = timer()
        logger.info(
            f"transform_one_hot_encodings: df: {self.df_train.shape} time taken {end_main - start_main}"
        )

    #######################
    ### Feature Importance Filter
    #######################
    def _feature_importance_filter_00009999(self, drop_cols_fi_0009999=None):
        """
        Filters out features from the training and test datasets that have an importance less than 0.00009999.

        Parameters:
        drop_cols_fi_0009999 (list, optional): List of columns to drop. If None, it calculates which features to drop based on their importance.
        """
        logger.info("Feature-Importance-Filter-0.0009999")
        feature_selection = FeatureSelection()

        if drop_cols_fi_0009999 == None:
            drop_cols_fi_0009999 = feature_selection.drop_unimportant_features_fit(
                self.docu_path,
                0.0009999,
                self.df_train_working_features,
                self.df_train_working_target,
                self.alt_config,
            )
            logger.info(f"Dropping {len(drop_cols_fi_0009999)} columns")

            self.export_list.append(dict(action_num=11, params=drop_cols_fi_0009999))

        self.df_train = feature_selection.drop_unimportant_features_transform(
            self.df_train, drop_cols_fi_0009999
        )
        self.df_test = feature_selection.drop_unimportant_features_transform(
            self.df_test, drop_cols_fi_0009999
        )
        self.df_train_working_features = (
            feature_selection.drop_unimportant_features_transform(
                self.df_train_working_features, drop_cols_fi_0009999
            )
        )

    #######################
    ### Feature Importance Filter
    #######################
    def _feature_importance_filter_00049999(
        self, drop_cols_fi_00049999=None, alt_config=None
    ):
        """
        Filters out features from the training and test datasets that have an importance less than 0.00049999.

        Parameters:
        drop_cols_fi_00049999 (list, optional): List of columns to drop. If None, it calculates which features to drop based on their importance.
        alt_config (dict, optional): Alternative configuration settings if provided.
        """
        feature_selection = FeatureSelection()

        if drop_cols_fi_00049999 == None:
            drop_cols_fi_00049999 = feature_selection.drop_unimportant_features_fit(
                self.docu_path,
                0.00049999,
                self.df_train_working_features,
                self.df_train_working_target,
                alt_config,
            )
            logger.info(f"Dropping {len(drop_cols_fi_00049999)} columns")

            self.export_list.append(dict(action_num=12, params=drop_cols_fi_00049999))

        self.df_train = feature_selection.drop_unimportant_features_transform(
            self.df_train, drop_cols_fi_00049999
        )
        self.df_test = feature_selection.drop_unimportant_features_transform(
            self.df_test, drop_cols_fi_00049999
        )
        self.df_train_working_features = (
            feature_selection.drop_unimportant_features_transform(
                self.df_train_working_features, drop_cols_fi_00049999
            )
        )

    #######################
    #### PCA
    #######################
    def _pca(self, pca_regr=None):
        """
        Applies Principal Component Analysis (PCA) transformation to reduce dimensionality of the feature sets.

        Parameters:
        pca_regr (PCA object, optional): Pre-fitted PCA transformer. If None, a new PCA fitting is performed.
        """
        feature_extraction = FeatureExtraction()

        if pca_regr == None:
            if "timestamps" in self.df_train_working_features:
                self.df_train_working_features.drop(
                    columns=["timestamps"], inplace=True
                )
            pca_regr = feature_extraction.pca_fit(
                self.df_train_working_features, 0.9999
            )
            self.export_list.append(dict(action_num=13, params=pca_regr))

        self.df_test_working = self.df_test.copy()
        self.df_test_working = self.clear_df_train(
            self.df_test_working, clear_cols=self.unrelated_cols
        )
        self.df_test_working_features, self.df_test_working_target = get_feature_df(
            self.df_test_working, target_names_list=self.target_names_list
        )

        transformed_features_train = feature_extraction.pca_transform(
            self.df_train_working_features, pca_regr
        )
        transformed_features_test = feature_extraction.pca_transform(
            self.df_test_working_features, pca_regr
        )
        transformed_features_train_working = feature_extraction.pca_transform(
            self.df_train_working_features, pca_regr
        )

        self.df_train = pd.concat(
            [transformed_features_train, self.df_train_working_target], axis=1
        )
        self.df_test = pd.concat(
            [transformed_features_test, self.df_test_working_target], axis=1
        )
        self.df_train_working_features = transformed_features_train_working

    #######################
    ### Polynominal Features
    #######################
    def _polynominal_features(self, poly_dict=None):
        """
        Enhances feature set by creating polynomial and interaction terms.

        Parameters:
        - poly_dict (dict, optional): Contains pre-fitted polynomial regressor and list of float columns. If None, fits a new polynomial transformer.
        """
        feature_interactions = FeatureInteractions()
        logger.info(
            "df_train, df_test, df_train_working before transform_polynominal_interaction: "
            "df_train shape: %s, df_test shape: %s, df_train_working_features shape: %s",
            self.df_train.shape,
            self.df_test.shape,
            self.df_train_working_features.shape,
        )
        if poly_dict == None:
            (
                poly_regr,
                poly_float_columns,
            ) = feature_interactions.polynominal_interaction_fit(
                self.df_train_working_features, self.target_names_list
            )
            poly_dict = [poly_regr, poly_float_columns]
            self.export_list.append(
                dict(
                    action_num=14,
                    params=[
                        poly_regr,
                        poly_float_columns,
                    ],
                )
            )
        poly_regr = poly_dict[0]
        poly_float_columns = poly_dict[1]

        self.df_train = feature_interactions.polynominal_interaction_transform(
            self.df_train, poly_regr=poly_regr, float_columns=poly_float_columns
        )

        self.df_test = feature_interactions.polynominal_interaction_transform(
            self.df_test, poly_regr=poly_regr, float_columns=poly_float_columns
        )

        self.df_train_working_features = (
            feature_interactions.polynominal_interaction_transform(
                self.df_train_working_features,
                poly_regr=poly_regr,
                float_columns=poly_float_columns,
            )
        )
        logger.info(
            "transform_polynominal_interaction: df_train shape: %s, df_test shape: %s, df_train_working_features shape: %s",
            *(
                self.df_train.shape,
                self.df_test.shape,
                self.df_train_working_features.shape,
            ),
        )
        logger.info(self.df_train_working_features.columns.values)


'''
    #######################
    #### MinMax Scaling
    #######################
    def _minmax_scaling(self, minmax_regr=None):
        """
        Applies Min-Max scaling to the training, training feature, and test datasets using a MinMaxScaler object from sklearn. 
        If the scaler is not provided, it fits a new scaler to the training feature set and applies this scaler across all sets.

        Parameters:
        minmax_regr (MinMaxScaler, optional): A pre-fitted MinMaxScaler. If None, a new scaler is fitted using the training feature dataset.
        """
        feature_scaling = FeatureScaling()
        if minmax_regr == None:
            minmax_regr = feature_scaling.minmax_scaler_fit(self.df_train_working_features)
            self.export_list.append(dict(action_num=15, params=minmax_regr))
        self.df_train = feature_scaling.minmax_scaler_transform(self.df_train, minmax_regr)
        self.df_train_working_features = feature_scaling.minmax_scaler_transform(
            self.df_train_features, minmax_regr
        )
        self.df_test = feature_scaling.minmax_scaler_transform(
            self.df_test, minmax_regr
        )

    #######################
    #### Standard Scaling
    #######################
    def _standard_scaling(self, standard_regr=None):
        """
        Applies Standard Scaling to normalize the training, training features, and test datasets using a StandardScaler.
        If the scaler is not provided, a new scaler is fitted to the training dataset and applied uniformly.

        Parameters:
        standard_regr (StandardScaler, optional): A pre-fitted StandardScaler. If None, a new scaler is fitted using the training dataset.
        """
        feature_scaling = FeatureScaling()
        if standard_regr == None:
            standard_regr = feature_scaling.standard_scaler_fit(self.df_train)
            self.export_list.append(dict(action_num=16, params=standard_regr))
        self.df_train = feature_scaling.standard_scaler_transform(self.df_train, standard_regr)
        self.df_train_features = feature_scaling.standard_scaler_transform(
            self.df_train_features, standard_regr
        )
        self.df_test = feature_scaling.standard_scaler_transform(
            self.df_test, standard_regr
        )

    #######################
    #### Robust Scaling
    #######################
    def _robust_scaling(self, robust_regr):
        """
        Applies Robust Scaling to the training, training features, and test datasets to reduce the effects of outliers using a RobustScaler.
        If the scaler is not provided, a new scaler is fitted to the training dataset.

        Parameters:
        - robust_regr (RobustScaler, optional): A pre-fitted RobustScaler. If None, a new scaler is fitted using the training dataset.
        """
        feature_scaling = FeatureScaling()
        if robust_regr == None:
            robust_regr = feature_scaling.robust_scaler_fit(self.df_train)
            self.export_list.append(dict(action_num=17, params=robust_regr))
        self.df_train = feature_scaling.robust_scaler_transform(self.df_train, robust_regr)
        self.df_train_features = feature_scaling.robust_scaler_transform(
            self.df_train_features, robust_regr
        )
        self.df_test = feature_scaling.robust_scaler_transform(
            self.df_test, robust_regr
        )
'''
