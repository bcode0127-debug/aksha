"""Isolation Forest detector over the Mission2 feature table.

Pure numpy/pandas/scikit-learn/PyOD. No `google.*`, ADK or vertexai imports
(ADR-002).

The detector is fitted on the training split only. That split is purged of
every anomaly- and rare-event-labelled window by the adapter, so it is nominal
by construction.

CONTAMINATION -- ours, and it does less than it looks like it does.

PyOD requires a `contamination` value, documented as the expected proportion of
outliers in the training data. That assumption does not apply here: the
training split is nominal by construction, so the true proportion is zero, and
zero is not an allowed value. Verified in-session against pyod 3.6.5 /
scikit-learn 1.8.0, `contamination` does **not** affect the fitted trees at all:
`score_samples` is identical between `contamination=0.01` and `contamination=0.40`
for the same `random_state`. It only shifts `decision_function` by a constant,
via scikit-learn's `offset_`, which leaves the score *ranking* exactly
unchanged. Since the conformal p-value depends only on a score's rank against
the calibration scores, contamination cannot move it.

So CONTAMINATION is set to 0.01 rather than PyOD's 0.1 default purely so that
`threshold_`/`labels_` -- which this codebase never reads -- do not imply that a
tenth of the nominal training data is anomalous. The operating threshold comes
from split conformal calibration (`aksha_core.conformal.split`), not from PyOD.

OTHER CHOICES -- ours:

  * One global model across all 11 channels, with `channel` excluded from the
    feature matrix. The model therefore learns the union of nominal behaviour
    over the channels rather than a per-channel norm. Limitation worth naming:
    a window that is unremarkable for one channel but unusual for another may
    not be separated, since nothing in the feature vector says which channel it
    came from. `mahalanobis` carries some cross-channel context, but it is not
    a substitute for per-channel models.

  * N_ESTIMATORS = 200. ESA-ADB configures its own subsequence Isolation Forest
    with 200 trees; matching it keeps the comparison honest, but the value is
    ours for this feature table, not inherited from a published protocol.

  * Missing `seconds_since_last_tc` (11 training rows, windows preceding the
    first telecommand on record) is imputed with the maximum seen in training --
    a finite stand-in for "longer ago than anything observed". The imputation
    value is fitted on train and stored in the artifact, so calibration and
    test never influence it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

IDENTITY_COLUMNS: tuple[str, ...] = ("window_start", "channel", "label", "split")

CONTAMINATION = 0.01
N_ESTIMATORS = 200
RANDOM_STATE = 20260818


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns in table order: everything that is not identity."""
    return [c for c in frame.columns if c not in IDENTITY_COLUMNS]


@dataclass
class FittedDetector:
    """A fitted scaler + Isolation Forest, plus the column order they expect.

    `feature_columns` is part of the fitted state, not a convention: a feature
    matrix assembled in a different order is silently wrong rather than loudly
    broken, so the order travels with the model and is enforced on every call.
    """

    feature_columns: list[str]
    impute_values: dict[str, float]
    scaler: StandardScaler
    model: IForest
    n_train_rows: int = 0
    params: dict = field(default_factory=dict)

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Feature matrix in the fitted column order, with fitted imputation.

        Raises on any missing column rather than reindexing them into NaN.
        """
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(
                f"feature columns missing from input: {missing}. "
                f"detector expects exactly {self.feature_columns}"
            )
        ordered = frame.loc[:, self.feature_columns].copy()
        for column, value in self.impute_values.items():
            if column in ordered.columns:
                ordered[column] = ordered[column].fillna(value)
        return ordered.to_numpy(dtype="float64")

    def raw_scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Isolation Forest decision scores. Higher means more outlying -- this
        is PyOD's direction, verified in-session, and it is used directly as the
        nonconformity score by the conformal layer.
        """
        return self.model.decision_function(self.scaler.transform(self.matrix(frame)))


def fit_detector(
    train: pd.DataFrame,
    columns: list[str] | None = None,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> FittedDetector:
    """Fit imputation, scaling and the Isolation Forest on the training split.

    Everything fitted here sees training rows only. The calibration and test
    splits are transform-only for the whole life of the artifact; that is what
    makes the conformal guarantee in `aksha_core.conformal.split` meaningful.
    """
    columns = columns or feature_columns(train)
    frame = train.loc[:, columns]

    impute_values = {
        column: float(frame[column].max())
        for column in columns
        if frame[column].isna().any()
    }
    filled = frame.fillna(value=impute_values) if impute_values else frame
    raw = filled.to_numpy(dtype="float64")

    scaler = StandardScaler().fit(raw)
    model = IForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    ).fit(scaler.transform(raw))

    return FittedDetector(
        feature_columns=list(columns),
        impute_values=impute_values,
        scaler=scaler,
        model=model,
        n_train_rows=int(frame.shape[0]),
        params={
            "contamination": contamination,
            "n_estimators": n_estimators,
            "random_state": random_state,
        },
    )
