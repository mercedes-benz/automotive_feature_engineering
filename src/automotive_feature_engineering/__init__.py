# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from automotive_feature_engineering.main_feature_engineering import (
    FeatureEngineering,
)


def run_main(df_train, df_test, model, target_names_list, method_list, **kwargs):
    feature = FeatureEngineering(df_train, df_test, model, target_names_list, **kwargs)
    return feature.main(method_list)


def static(df_train, df_test, model, target_names_list, **kwargs):
    return run_main(
        df_train, df_test, model, target_names_list, method_list=None, **kwargs
    )


def manual(method_list, df_train, df_test, model, target_names_list, **kwargs):
    return run_main(df_train, df_test, model, target_names_list, method_list, **kwargs)


def rl(
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
    from automotive_feature_engineering.reinforcement_learning.rl_main import (
        ReinforcementLearning,
    )

    rl_instance = ReinforcementLearning(
        df_train,
        df_train_origin,
        df_test_origin,
        target_names_list,
        model,
        rl_raster,
        unrelated_cols,
        alt_config,
        alt_docu,
    )
    return rl_instance.rl_training()
