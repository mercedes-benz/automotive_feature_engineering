# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
import pandas as pd
from typing import Optional, Union
from joblib import dump, load
import ray
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import PartialAlgorithmConfigDict
from ray.tune import PlacementGroupFactory
from ray.tune.logger import pretty_print
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
from ray.tune.logger import pretty_print
from src.automotive_feature_engineering.reinforcement_learning.rl_environment_ss import (
    EnergaizeEnv2,
)

# from config import Config
import numpy as np
from src.automotive_feature_engineering.utils import data_loader_test
from src.automotive_feature_engineering.__init__ import manual


os.environ["OPENBLAS_NUM_THREADS"] = "1"


# The modified Algorithm class we will use:
# Subclassing from PPO, our algo will only modity `default_resource_request`,
# telling Ray Tune that it's ok (not mandatory) to place our n remote envs on a
# different node (each env using 1 CPU).
class PPORemoteInference(PPO):
    @classmethod
    @override(Algorithm)
    def default_resource_request(
        cls,
        config: Union[AlgorithmConfig, PartialAlgorithmConfigDict],
    ):
        if isinstance(config, AlgorithmConfig):
            cf = config
        else:
            cf = cls.get_default_config().update_from_dict(config)

        # Return PlacementGroupFactory containing all needed resources
        # (already properly defined as device bundles).
        return PlacementGroupFactory(
            bundles=[
                {
                    # Single CPU for the local worker. This CPU will host the
                    # main model in this example (num_workers=0).
                    "CPU": 1,
                    # Possibly add n GPUs to this.
                    "GPU": cf.num_gpus,
                },
                {
                    # Different bundle (meaning: possibly different node)
                    # for your n "remote" envs (set remote_worker_envs=True).
                    "CPU": cf.num_envs_per_worker,
                },
            ],
            strategy=cf.placement_strategy,
        )


class ReinforcementLearning:
    def __init__(
        self,
        df_train,
        df_train_origin,
        df_test_origin,
        target_names_list,
        model,
        rl_raster,
        unrelated_cols,
        alt_config,
        alt_docu,
    ):
        # self.energaize_config = energaize_config
        self.df_train = df_train
        self.df_train_origin = df_train_origin
        self.df_test_origin = df_test_origin
        self.target_names_list = target_names_list
        self.model = model
        # self.fuse = fuse
        # self.path = path
        self.rl_raster = rl_raster
        self.unrelated_cols = unrelated_cols
        self.alt_config = alt_config
        self.alt_docu = alt_docu

    def rl_training(self):
        # energaize_config = Config.getConfig()

        env_conf = {
            # "energaize_config": energaize_config,
            "df_train": self.df_train,
            "target_names_list": self.target_names_list,
            # "path": self.path,
            # "fuse": self.fuse,
            "unrelated_cols": self.unrelated_cols,
            "alt_config": self.alt_config,
            "alt_docu": self.alt_docu,
            "rl_raster": self.rl_raster,
        }
        print("Starting reinforcement learning training...")

        ray.init(local_mode=False, ignore_reinit_error=True)

        register_env("multienv", lambda env_config: EnergaizeEnv2(env_config))

        # main part: configure the ActionMaskEnv and ActionMaskModel
        config = (
            ppo.PPOConfig()
            .environment(
                # random env with 100 discrete actions and 5x [-1,1] observations
                # some actions are declared invalid and lead to errors
                env="multienv",
                env_config=env_conf,
                disable_env_checking=True,
            )
            .rollouts(
                restart_failed_sub_environments=True,
                ignore_worker_failures=True,
                # Force sub-envs to be ray.actor.ActorHandles, so we can step
                # through them in parallel.
                remote_worker_envs=True,
                num_envs_per_worker=14,
                # Use a single worker (however, with n parallelized remote envs, maybe
                # even running on another node).
                # Action computations will occur on the "main" (GPU?) node, while
                # the envs run on one or more CPU node(s).
                num_rollout_workers=0,
                recreate_failed_workers=True,
            )
            .training(
                # the ActionMaskModel retrieves the invalid actions and avoids them
                model={
                    "custom_model": ActionMaskModel,
                    # disable action masking according to CLI
                    "custom_model_config": {"no_masking": False},
                },
                train_batch_size=128,
            )
            .framework("tf2", eager_tracing=False, eager_max_retraces=None)
            .resources(
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                # Set the number of CPUs used by the (local) worker, aka "driver"
                # to match the number of ray remote envs.
                num_cpus_for_local_worker=14 + 1,
            )
        )

        # manual training loop using PPO and manually keeping track of state
        # algo = PPORemoteInference(config=config)
        algo = config.build()

        # run manual training loop and print results after each iteration
        for i in range(1):
            result = algo.train()
            print(pretty_print(result))

        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

        env = EnergaizeEnv2(env_conf)
        num_episodes = 1  # Number of episodes

        all_episode_actions = []  # List to store actions for all episodes

        for _ in range(num_episodes):
            terminated = False
            episode_actions = []  # List to store actions for the current episode
            observations, _ = env.reset()

            while not terminated:
                action = algo.compute_single_action(observations)
                observations, reward, terminated, truncated, info = env.step(action)
                episode_actions.append(action)  # Append action to the current episode

            all_episode_actions.append(
                episode_actions
            )  # Append episode actions to the list

        ray.shutdown()
        method_list = all_episode_actions[0]
        results = manual(
            method_list,
            self.df_train_origin,
            self.df_test_origin,
            self.model,
            self.target_names_list,
            unrelated_cols=self.unrelated_cols,
            alt_config=self.alt_config,
            alt_docu_path=self.alt_docu,
        )

        return results
