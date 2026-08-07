# SPDX-License-Identifier: MIT
"""Regression tests for feature-importance filter fallback (issue #6)."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import pandas as pd
import sklearn.utils.metaestimators as _metaestimators

# eli5 0.13 still imports the removed sklearn helper; provide a shim so the
# package can be imported on newer scikit-learn versions.
if not hasattr(_metaestimators, "if_delegate_has_method"):
    from sklearn.utils.metaestimators import available_if

    def if_delegate_has_method(delegate):  # type: ignore[misc]
        return available_if(lambda self: hasattr(self, delegate))

    _metaestimators.if_delegate_has_method = if_delegate_has_method
    sys.modules["sklearn.utils.metaestimators"] = _metaestimators

from automotive_feature_engineering.feature_selection import FeatureSelection


class TestFeatureImportanceFilterFallback(unittest.TestCase):
    def test_keeps_features_when_all_below_threshold(self) -> None:
        """If every feature is below the threshold, keep all instead of emptying."""
        cols = [f"sig_{i}" for i in range(8)]
        df_features = pd.DataFrame({c: [0, 1, 0, 1] for c in cols})
        df_target = pd.DataFrame({"target": [0, 1, 0, 1]})
        # Near-zero importances for every column (typical of uniform/one-hot CAN signals).
        fake_importances = [(0.0, c) for c in cols]

        fs = FeatureSelection()
        with patch.object(
            FeatureSelection,
            "calc_globalFeatureImportance",
            return_value=fake_importances,
        ):
            drop_cols = fs.drop_unimportant_features_fit(
                ".", 0.0009999, df_features, df_target
            )

        self.assertEqual(drop_cols, [])
        transformed = fs.drop_unimportant_features_transform(df_features, drop_cols)
        self.assertEqual(list(transformed.columns), list(df_features.columns))

    def test_still_drops_when_some_features_pass(self) -> None:
        cols = ["keep_me", "drop_me"]
        df_features = pd.DataFrame({c: [0, 1, 0, 1] for c in cols})
        df_target = pd.DataFrame({"target": [0, 1, 0, 1]})
        fake_importances = [(0.5, "keep_me"), (0.0, "drop_me")]

        fs = FeatureSelection()
        with patch.object(
            FeatureSelection,
            "calc_globalFeatureImportance",
            return_value=fake_importances,
        ):
            drop_cols = fs.drop_unimportant_features_fit(
                ".", 0.0009999, df_features, df_target
            )

        self.assertEqual(list(drop_cols), ["drop_me"])
        transformed = fs.drop_unimportant_features_transform(df_features, drop_cols)
        self.assertEqual(list(transformed.columns), ["keep_me"])

    def test_transform_does_not_empty_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        fs = FeatureSelection()
        # Simulate a bad drop list that would remove every column.
        out = fs.drop_unimportant_features_transform(df, ["a", "b"])
        self.assertEqual(list(out.columns), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
