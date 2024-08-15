# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


class FeatureExtraction:
    """
    This class provides methods for reducing dimensionality of datasets using feature extraction
    techniques such as Independent Component Analysis (ICA) and Principal Component Analysis (PCA).
    """

    ##########################################
    ### ICA
    ##########################################

    def ica_fit(self, df: pd.DataFrame) -> object:
        """
        Fits an Independent Component Analysis (ICA) model to the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing features for ICA.

        Returns:
        - object: A fitted ICA model.
        """
        ica_regr = FastICA(tol=0.01, whiten_solver="eigh", max_iter=500)
        ica_regr.fit(df)

        return ica_regr

    def transform_ica(self, df: pd.DataFrame, ica_regr: object) -> pd.DataFrame:
        """
        Applies the ICA transformation to the DataFrame using the provided ICA model.

        Parameters:
        - df (pd.DataFrame): DataFrame to transform.
        - ica_regr (object): Fitted ICA model.

        Returns:
        - pd.DataFrame: DataFrame containing the independent components as features.
        """
        df_ica_features = ica_regr.transform(df)
        n_rows, n_cols = df_ica_features.shape
        df = pd.DataFrame(
            df_ica_features,
            columns=["ICA%i" % i for i in range(n_cols)],
            index=df.index,
        )

        return df

    ##########################################
    ### PCA
    ##########################################

    def pca_fit(self, df: pd.DataFrame, n_components: float) -> object:
        """
        Fits a Principal Component Analysis (PCA) model to the DataFrame with the specified number
        of components.

        Parameters:
        - df (pd.DataFrame): DataFrame containing features for PCA.
        - n_components (float): The number of principal components to compute.

        Returns:
        - object: A fitted PCA model.
        """
        pca_regr = PCA(n_components=n_components)
        pca_regr.fit(df)
        return pca_regr

    def pca_transform(self, df: pd.DataFrame, pca_regr: object) -> pd.DataFrame:
        """
        Applies the PCA transformation to the DataFrame using the provided PCA model.

        Parameters:
        - df (pd.DataFrame): DataFrame to transform.
        - pca_regr (object): Fitted PCA model.

        Returns:
        - pd.DataFrame: DataFrame containing the principal components as features.
        """
        # Use features that only in fit
        df_pca_features = pca_regr.transform(df)
        n_rows, n_cols = df_pca_features.shape
        df = pd.DataFrame(
            df_pca_features,
            columns=["PCA%i" % i for i in range(n_cols)],
            index=df.index,
        )

        return df
