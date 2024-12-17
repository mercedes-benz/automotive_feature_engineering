# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import gymnasium as gym
import os
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
import logging
from typing import Optional
import math
import pathlib
from automotive_feature_engineering.sna_handling import SnaHandling
from automotive_feature_engineering.feature_extraction import FeatureExtraction
from automotive_feature_engineering.feature_encoding import FeatureEncoding
from automotive_feature_engineering.feature_selection import FeatureSelection
from automotive_feature_engineering.feature_scaling import FeatureScaling
from automotive_feature_engineering.feature_interactions import FeatureInteractions

from automotive_feature_engineering.utils.utils import get_feature_df
from sklearn.model_selection import train_test_split
import automotive_feature_engineering.utils.utils as utils

# from ray.rllib import agents
from ray.rllib.utils import try_import_tf

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

tf = try_import_tf()

logger = logging.getLogger(__name__)

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


class EnergaizeEnv2(gym.Env):
    def __init__(self, env_config) -> None:

        ### Training and Test Data
        # self.df_list = env_config["df"]
        # self.df, self.df_train = self.draw_df(self.df_list)
        self.rl_raster = env_config["rl_raster"]
        self.df_train = env_config["df_train"]
        self.target_names_list = env_config["target_names_list"]
        self.unrelated_cols = env_config[
            "unrelated_cols"
        ]  # Columns that are not part of the raw data
        self.df_train = self.clear_df_train(
            self.df_train, clear_cols=self.unrelated_cols
        )
        self.alt_config = env_config["alt_config"]
        self.alt_docu = env_config["alt_docu"]
        if self.alt_docu == None:
            self.alt_docu = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.alt_docu = self.alt_docu
            # check if the path exists, if not create it
            if not os.path.exists(self.docu_path):
                os.makedirs(self.docu_path)
        # self.fuse_prefix = "I_"
        # self.path = env_config["path"]
        # self.fuse = env_config["fuse"]

        self.df_train_X_train = None
        self.df_train_X_test = None
        self.df_train_y_train = None
        self.df_train_y_test = None

        ### Actions we can take, function zero -> 0, function a -> 1, function b -> 2, function c -> 3
        self.num_actions = 15
        # self._skip_env_checking = True

        ### Set max sequence length
        self.max_episode_steps = self.num_actions - 7
        ### Set current sequence length 0
        self.current_sequence_length = 0

        # Masking only works for Discrete actions.
        self.state = spaces.Box(
            0, self.num_actions - 1, shape=(self.max_episode_steps,), dtype=int
        )

        self.action_space = spaces.Discrete(self.num_actions)

        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(0, 1, shape=(self.num_actions,), dtype=int),
                "observations": spaces.Box(
                    0,
                    self.num_actions - 1,
                    shape=(self.max_episode_steps,),
                    dtype=int,
                ),
            }
        )
        self.total_steps = 0

        ### Methods
        self.feature_selection = FeatureSelection()
        self.sna_handling = SnaHandling()
        self.feature_scaling = FeatureScaling()
        self.feature_interactions = FeatureInteractions()
        self.feature_encoding = FeatureEncoding()
        self.feature_extraction = FeatureExtraction()

        self.reset()

    ##########################################
    # Step
    ##########################################
    def step(
        self, action: int
    ) -> tuple[dict[np.array, np.array], float, bool, bool, dict]:
        ### Increase sequence length
        self.current_sequence_length += 1

        ### Set placeholder for info
        infos = {}

        # True if environment terminates (eg. due to task completion, failure etc.)
        terminated = False
        # True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP
        truncated = False

        self.state[self.current_sequence_length - 1] = action
        # valid actions
        self.action_mask = [
            int(action not in self.state) for action in list(range(self.num_actions))
        ]

        # containing sns method index
        index_list = [3, 4, 5, 6, 7, 8]
        # Check if any element is 0
        if any(self.action_mask[i] == 0 for i in index_list):
            for i in index_list:
                self.action_mask[i] = 0

        self.action_mask[0] = 1
        # Feature importance always available
        self.action_mask[11] = 1
        self.action_mask[12] = 1

        # poly features not possible if df too large
        if len(self.df_train_X_train.columns) > 200:
            self.action_mask[14] = 0
        elif len(self.df_train_X_train.columns) <= 200 and 14 not in self.state:
            self.action_mask[14] = 1

        obs = {
            "action_mask": np.asarray(self.action_mask),
            "observations": self.state,
        }

        ### Take action
        try:
            self.total_steps += 1
            self._take_action(action)

            if self.df_train_X_train.shape[1] > 20000:
                reward = -1
                terminated = True
                return obs, reward, terminated, truncated, infos
        except Exception as e:
            print("STATE ACTION EXC", self.state, e)
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, infos

        # poly features not possible if df too large
        if len(self.df_train_X_train.columns) > 200:
            self.action_mask[14] = 0
        elif len(self.df_train_X_train.columns) <= 200 and 14 not in self.state:
            self.action_mask[14] = 1

        obs = {
            "action_mask": np.asarray(self.action_mask),
            "observations": self.state,
        }

        ### Check if maximum sequence length is reached
        ### Calculate reward
        if self.current_sequence_length >= self.max_episode_steps:
            if self.df_train_X_train.shape[1] > 1000:
                reward = -1
                terminated = True
                return obs, reward, terminated, truncated, infos
            try:
                reward = self._calculate_performance()
                terminated = True
                print("reward: ", reward)
                return obs, reward, terminated, truncated, infos
            except Exception as e:
                print("error", e)
                reward = -1
                terminated = True
                return obs, reward, terminated, truncated, infos
        else:
            reward = 0

        ### Return step information
        return (
            obs,
            reward,
            terminated,
            truncated,
            infos,
        )

    ##########################################
    # (Render)
    ##########################################
    def render(self):
        # No vizualisation
        pass

    ##########################################
    # Reset
    ##########################################
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.array, dict]:
        super().reset(seed=seed)

        self.action_mask = np.zeros((self.num_actions,), dtype=int)

        # containing sns method index
        index_list = [3, 4, 5, 6, 7, 8]
        # Check if any element is 0
        for i in index_list:
            self.action_mask[i] = 1

        self.state = np.zeros((self.max_episode_steps,), dtype=int)
        obs = {
            "action_mask": self.action_mask,
            "observations": self.state,
        }

        ### Training and Test Data
        # self.df, self.df_train = self.draw_df(self.df_list)
        self.df_train_X, self.df_train_y = get_feature_df(
            self.df_train, target_names_list=self.target_names_list
        )
        (
            self.df_train_X_train,
            self.df_train_X_test,
            self.df_train_y_train,
            self.df_train_y_test,
        ) = train_test_split(
            self.df_train_X, self.df_train_y, test_size=0.2, random_state=42
        )
        print(f"df_train_X_train shape: {self.df_train_X_train.shape}")
        print(f"df_train_X_test shape: {self.df_train_X_test.shape}")
        print(f"df_train_y_train shape: {self.df_train_y_train.shape}")
        print(f"df_train_y_test shape: {self.df_train_y_test.shape}")

        # self.df_test_working = self.df_test.copy()

        # self.df_train_working_features, self.df_train_working_target = get_feature_df(self.df_train_working, target_names_list=self.target_names_list)

        print("IN RESET")

        # ### Remove "file" and "I_" before processing
        # self.df, self.df_target = get_feature_df(self.df, self.fuse_prefix)
        # self.df_train, self.df_train_target = get_feature_df(
        #     self.df_train, self.fuse_prefix
        # )

        ### Reset sequence length
        self.current_sequence_length = 0

        ### Set placeholder for info
        infos = {}

        return obs, infos

    ##########################################
    ### Take Action

    # 1 -> drop correlated signals 0.9

    ##########################################
    def _take_action(self, action: int) -> None:
        if action == 0:
            print("Placeholder")
        # 0 -> remove highly correlated features
        elif action == 1:
            print(f"Take Action {action}")
            feature_selection = FeatureSelection()

            importances = self.feature_selection.calc_globalFeatureImportance(
                self.alt_docu,
                "randomforest",
                self.df_train_X_train,
                self.df_train_y_train,
                self.alt_config,
            )
            # ### Remove "file" and "I_" before processing
            # self.df, self.df_target = get_feature_df(
            #     self.df, fuse_prefix=self.fuse_prefix
            # )
            # self.df_train, self.df_train_target = get_feature_df(
            #     self.df_train, self.fuse_prefix
            # )

            drop_cols_corr = self.feature_selection.drop_correlated_features_fit(
                self.df_train_X_train,
                0.9,
                importances,
            )

            # self.df = self.feature_selection.drop_correlated_features_transform(
            #     self.df, drop_cols_corr
            # )

            self.df_train_X_train = (
                self.feature_selection.drop_correlated_features_transform(
                    self.df_train_X_train, drop_cols_corr
                )
            )

            self.df_train_X_test = (
                self.feature_selection.drop_correlated_features_transform(
                    self.df_train_X_test, drop_cols_corr
                )
            )
            # exit(1)

        # 1 -> remove highly correlated features
        elif action == 2:
            print(f"Take Action {action}")
            # self.df = combine_dfs([self.df, self.df_target])
            # self.df_train = combine_dfs([self.df_train, self.df_train_target])
            importances = self.feature_selection.calc_globalFeatureImportance(
                self.alt_docu,
                "randomforest",
                self.df_train_X_train,
                self.df_train_y_train,
                self.alt_config,
            )
            # ### Remove "file" and "I_" before processing
            # self.df, self.df_target = get_feature_df(
            #     self.df, fuse_prefix=self.fuse_prefix
            # )
            # self.df_train, self.df_train_target = get_feature_df(
            #     self.df_train, self.fuse_prefix
            # )

            drop_cols_corr = self.feature_selection.drop_correlated_features_fit(
                self.df_train_X_train,
                0.95,
                importances,
            )

            self.df_train_X_train = (
                self.feature_selection.drop_correlated_features_transform(
                    self.df_train_X_train, drop_cols_corr
                )
            )

            self.df_train_X_test = (
                self.feature_selection.drop_correlated_features_transform(
                    self.df_train_X_test, drop_cols_corr
                )
            )

        # 2 -> sns median 8
        elif action == 3:
            print(f"Take Action {action}")
            self.df_train_X_train = self.sna_handling.fill_sna_median_rl(
                self.df_train_X_train, 8
            )

            self.df_test_X_test = self.sna_handling.fill_sna_median_rl(
                self.df_train_X_test, 8
            )

        # 3 -> sns median 32
        elif action == 4:
            print(f"Take Action {action}")
            self.df_train_X_train = self.sna_handling.fill_sna_median_rl(
                self.df_train_X_train, 32
            )

            self.df_test_X_test = self.sna_handling.fill_sna_median_rl(
                self.df_train_X_test, 32
            )

        # 4 -> sns mean 8
        elif action == 5:
            print(f"Take Action {action}")
            self.df_train_X_train = self.sna_handling.fill_sna_mean_rl(
                self.df_train_X_train, 8
            )

            self.df_train_X_test = self.sna_handling.fill_sna_mean_rl(
                self.df_train_X_test, 8
            )

        # 4 -> sns mean 32
        elif action == 6:
            print(f"Take Action {action}")
            self.df_train_X_train = self.sna_handling.fill_sna_mean_rl(
                self.df_train_X_train, 32
            )

            self.df_train_X_test = self.sna_handling.fill_sna_mean_rl(
                self.df_train_X_test, 32
            )

        # -> sns zero 8
        elif action == 7:
            print(f"Take Action {action}")
            self.df_train_X_train = self.sna_handling.fill_sna_zero_rl(
                self.df_train_X_train, 8
            )

            self.df_train_X_test = self.sna_handling.fill_sna_zero_rl(
                self.df_train_X_test, 8
            )

        # -> sns zero 32
        elif action == 8:
            print(f"Take Action {action}")
            self.df_train_X_train = self.sna_handling.fill_sna_zero_rl(
                self.df_train_X_train, 32
            )

            self.df_train_X_test = self.sna_handling.fill_sna_zero_rl(
                self.df_train_X_test, 32
            )

        elif action == 9:
            print(f"Take Action {action}")
            # self.df, self.df_train = self.feature_selection.filter_by_variance(
            #     self.df, self.df_train
            # )
            drop_col_var = self.feature_selection.filter_by_variance_fit(
                self.df_train_X_train, th=0.1
            )

            self.df_train_X_train = self.feature_selection.filter_by_variance_transform(
                self.df_train_X_train, drop_col_var
            )
            self.df_train_X_test = self.feature_selection.filter_by_variance_transform(
                self.df_train_X_test, drop_col_var
            )

        # OHE
        elif action == 10:
            print(f"Take Action {action}")
            ohe_regr = self.feature_encoding.one_hot_encoding_fit(self.df_train_X_train)
            self.df_train_X_train = self.feature_encoding.one_hot_encoding_transform(
                self.df_train_X_train, ohe_regr=ohe_regr
            )
            self.df_train_X_test = self.feature_encoding.one_hot_encoding_transform(
                self.df_train_X_test, ohe_regr=ohe_regr
            )

        # feature importance
        elif action == 11:
            print(f"Take Action {action}")
            # self.df = combine_dfs([self.df, self.df_target])
            # self.df_train = combine_dfs([self.df_train, self.df_train_target])
            importance = 0.0009999
            importances = self.feature_selection.calc_globalFeatureImportance(
                self.alt_docu,
                "randomforest",
                self.df_train_X_train,
                self.df_train_y_train,
                self.alt_config,
            )
            self.df_train_X_train = self.feature_selection.dropUnimportantFeatures(
                self.df_train_X_train, importances, importance
            )

            self.df_train_X_test = self.feature_selection.dropUnimportantFeatures(
                self.df_train_X_test, importances, importance
            )
            ### Remove "file" and "I_" before processing
            # self.df, self.df_target = get_feature_df(self.df, self.fuse_prefix)
            # self.df_train, self.df_train_target = get_feature_df(
            #     self.df_train, self.fuse_prefix
            # )

        # feature importance
        elif action == 12:
            print(f"Take Action {action}")
            # self.df = combine_dfs([self.df, self.df_target])
            # self.df_train = combine_dfs([self.df_train, self.df_train_target])

            importance = 0.0049999
            importances = self.feature_selection.calc_globalFeatureImportance(
                self.alt_docu,
                "randomforest",
                self.df_train_X_train,
                self.df_train_y_train,
                self.alt_config,
            )
            self.df_train_X_train = self.feature_selection.dropUnimportantFeatures(
                self.df_train_X_train, importances, importance
            )

            self.df_train_X_test = self.feature_selection.dropUnimportantFeatures(
                self.df_train_X_test, importances, importance
            )
            ### Remove "file" and "I_" before processing
            # self.df, self.df_target = get_feature_df(self.df, self.fuse_prefix)
            # self.df_train, self.df_train_target = get_feature_df(
            #     self.df_train, self.fuse_prefix
            # )

        ### PCA
        elif action == 13:
            print(f"Take Action {action}")
            ### debugging
            if "timestamps" in self.df_train_X_train:
                self.df_train_X_train.drop(columns=["timestamps"])
            ######
            pca_regr = self.feature_extraction.pca_fit(self.df_train_X_train, 0.95)
            self.df_train_X_train = self.feature_extraction.pca_transform(
                self.df_train_X_train, pca_regr
            )
            self.df_train_X_test = self.feature_extraction.pca_transform(
                self.df_train_X_test, pca_regr
            )

        ### Poly Features
        elif action == 14:
            print(f"Take Action {action}")
            self.df_train_X_train = (
                self.feature_interactions.fit_transform_polynominal_interaction_rl2(
                    self.df_train_X_train
                )
            )

            self.df_train_X_test = (
                self.feature_interactions.fit_transform_polynominal_interaction_rl2(
                    self.df_train_X_test
                )
            )

        elif action == 15:
            print(f"Take Action {action}")
            minmax_scaler = self.feature_scaling.minmax_scaler_fit(
                self.df_train_X_train
            )
            self.df_train_X_train = self.feature_scaling.minmax_scaler_transform(
                self.df_train_X_train, minmax_scaler
            )

            self.df_train_X_test = self.feature_scaling.minmax_scaler_transform(
                self.df_train_X_test, minmax_scaler
            )

        elif action == 16:
            print(f"Take Action {action}")
            standard_scaler = self.feature_scaling.standard_scaler_fit(
                self.df_train_X_train
            )
            self.df_train_X_train = self.feature_scaling.standard_scaler_transform(
                self.df_train_X_train, standard_scaler
            )
            self.df_test_X_test = self.feature_scaling.standard_scaler_transform(
                self.df_test_X_test, standard_scaler
            )

        elif action == 17:
            print(f"Take Action {action}")
            robust_scaler = self.feature_scaling.robust_scaler_fit(
                self.df_train_X_train
            )
            self.df_train_X_train = self.feature_scaling.robust_scaler_transform(
                self.df_train_X_train, robust_scaler
            )

            self.df_train_X_test = self.feature_scaling.robust_scaler_transform(
                self.df_train_X_test, robust_scaler
            )
        else:
            return "action not available"

    ##########################################
    # Calculate reward
    ##########################################
    def _calculate_performance(self) -> float:

        ### Add "file" and "I_" before processing
        # self.df = combine_dfs([self.df, self.df_target])

        ### Crossvalidation
        # cv = CrossValidation(self.df)
        # df_test = cv.create_testset()

        # splitList = cv.create_splits()

        # self.raster = 1.5 # default or manually adjustment check

        valR2 = []

        X_trn, y_trn = self.df_train_X_train, self.df_train_y_train
        X_val, y_val = self.df_train_X_test, self.df_train_y_test

        # for splitindex, split in enumerate(splitList):
        #     preprocessor = PreprocessData(split[2])
        #     labelcols = preprocessor.getLabelColumns()
        #     X_trn = preprocessor.getX()
        #     y_trn = preprocessor.gety()

        #     X_val = preprocessor.getX_val(split, self.df_test)
        #     y_val = preprocessor.gety_val(split, self.df_test)

        ### Train Model
        config_dict = utils.get_dict(
            config_path=self.alt_config,
            default_path=pathlib.Path(__file__).resolve().parent.parent.as_posix()
            + "/reinforcement_learning/rl_randomforest_defaults.json",
        )
        regr = RandomForestRegressor(**config_dict)
        regr.fit(X_trn, y_trn)

        pred_val = regr.predict(X_val).reshape(-1, 1)


        ### R2
        valR2.append(r2_score(y_val, pred_val, multioutput="raw_values"))
        print(f"valR2 {valR2}")
        ### empty list for storing the rewards
        rewards = []
        ### R2 average
        r2_avg = np.average(valR2)

        print("STATE CALC", self.state)

        transformed_r2 = self.expo_r2(r2_avg)
        rewards.append(transformed_r2)

        ### Signals
        if self.df_train_X_train.shape[1] > 400:
            rewards.append(-1)

        reward = np.mean(rewards)

        print(
            f"Rewards : {rewards} mean: {reward} signals {self.df_train_X_train.shape[1]}"
        )
        ### Return Score
        return reward

    ##########################################
    # Create dfs
    ##########################################
    # def draw_df(self, dataframes):
    #     # how many random dfs to draw
    #     # num_numbers = 2

    #     # index_numbers = random.sample(range(len(dataframes)), num_numbers)
    #     # dataframes_subset = [dataframes[index] for index in index_numbers]

    #     # df = pd.concat(dataframes, join="outer")
    #     df = dataframes

    #     cv = CrossValidation(df)
    #     # Create virtual test set
    #     df_test = (
    #         cv.create_testset()
    #     )  # Nummer der Fahrt oder Liste, die maximal so lang ist wie testDriveID zb Random Indices übergeben.

    #     # DF without Test Set --> NEVER touch test set for anything! Do all global XAI without test set, but apply them on entire data (df)!
    #     df_train = df_test[2].copy()
    #     df_train = df_train.drop(
    #         columns=["file", "timestamps"], errors="ignore", axis=1
    #     )
    #     return df, df_train

    def tanh_pawd(self, x):
        return (math.exp(x + 2) - math.exp(-x + 2)) / (
            math.exp(x + 2) + math.exp(-x + 2)
        )

    def tanh_r2(self, x):
        return (math.exp(x + 1) - math.exp(-x + 1)) / (
            math.exp(x + 1) + math.exp(-x + 1)
        )

    def relu_pawd(self, x):
        return max(-1, (20 + x + np.abs(20 + x)) / 20 - 1)

    def relu_r2(self, x):
        return max(-1, (x + np.abs(x) / 1 - 1))

    def relu_signals(self, x):
        return max(-1, (200 + x + np.abs(200 + x)) / 200 - 1)

    def expo_r2(self, x):
        return 2 ** (4 * x - 3) - 1

    def expo_pawd(self, x):
        return 2 ** (-0.2 * x + 1) - 1

    def calculate_mean(self, *args):
        return sum(args) / len(args)

    # Method to calculate the integral based on a time-interval and a series of values
    # ********************************************************
    # time [float] --> period between two datapoints
    # series [np.array] --> data which shall be integrated
    # ********************************************************
    def __calculate_integral(self, time, series):
        return time * np.cumsum(series)  # check

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
