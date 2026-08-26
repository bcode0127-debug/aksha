"""Tests for the IForest + split conformal detector.

Synthetic data throughout -- these run in CI without the Mission2 download or a
trained artifact.

The properties under test are the ones whose failure would be silent: leakage
of calibration/test data into the fit, a round-tripped artifact that scores
differently from the one that was saved, p-values outside [0,1], and a feature
matrix assembled in the wrong order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aksha_core.conformal.split import SplitConformalCalibrator
from aksha_core.detectors.artifact import DetectorArtifact
from aksha_core.detectors.iforest import IDENTITY_COLUMNS, feature_columns, fit_detector

FEATURES = ["alpha", "beta", "gamma"]


def _frame(n: int, split: str, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=shift, size=(n, len(FEATURES)))
    frame = pd.DataFrame(data, columns=FEATURES)
    frame["window_start"] = pd.date_range("2001-01-01", periods=n, freq="h")
    frame["channel"] = "channel_18"
    frame["label"] = "nominal"
    frame["split"] = split
    return frame


@pytest.fixture(scope="module")
def splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _frame(600, "train", seed=1),
        _frame(300, "calibration", seed=2),
        _frame(200, "test", seed=3, shift=0.4),
    )


@pytest.fixture(scope="module")
def artifact(splits) -> DetectorArtifact:
    train, calibration, _ = splits
    detector = fit_detector(train, columns=FEATURES)
    calibrator = SplitConformalCalibrator(detector.raw_scores(calibration))
    return DetectorArtifact(
        detector_version="test-0.0.1",
        detector=detector,
        calibrator=calibrator,
        metadata={},
    )


# --- identity columns are not features ----------------------------------------


def test_feature_columns_excludes_identity(splits):
    train, _, _ = splits
    assert set(feature_columns(train)) == set(FEATURES)
    for column in IDENTITY_COLUMNS:
        assert column not in feature_columns(train)


# --- the scaler and model never see calibration or test data ------------------


def test_scaler_statistics_come_from_train_alone(splits):
    """The scaler's fitted mean must equal the training split's mean exactly.

    If any calibration or test row had reached the fit, these would differ --
    the test split is deliberately shifted by 0.4 so leakage would be visible.
    """
    train, calibration, test = splits
    detector = fit_detector(train, columns=FEATURES)

    np.testing.assert_allclose(
        detector.scaler.mean_, train[FEATURES].to_numpy().mean(axis=0), rtol=1e-12
    )
    np.testing.assert_allclose(
        detector.scaler.var_, train[FEATURES].to_numpy().var(axis=0), rtol=1e-12
    )

    pooled = pd.concat([train, calibration, test])[FEATURES].to_numpy().mean(axis=0)
    assert not np.allclose(detector.scaler.mean_, pooled, rtol=1e-6), (
        "scaler mean matches the pooled mean -- calibration/test leaked into the fit"
    )


def test_fit_row_count_matches_train_only(splits):
    train, _, _ = splits
    detector = fit_detector(train, columns=FEATURES)
    assert detector.n_train_rows == len(train)


def test_imputation_value_is_fitted_on_train_only(splits):
    """A NaN in train is filled from train's own maximum, never from a pooled
    maximum that calibration or test could raise.
    """
    train, _, test = splits
    train = train.copy()
    train.loc[train.index[0], "alpha"] = np.nan
    test = test.copy()
    test["alpha"] = 1_000.0  # far above anything in train

    detector = fit_detector(train, columns=FEATURES)
    assert detector.impute_values["alpha"] == pytest.approx(train["alpha"].max())
    assert detector.impute_values["alpha"] < 1_000.0


# --- conformal p-values --------------------------------------------------------


def test_p_values_are_within_unit_interval(artifact, splits):
    _, _, test = splits
    p = artifact.calibrator.p_values(artifact.detector.raw_scores(test))
    assert p.min() >= 0.0
    assert p.max() <= 1.0


def test_p_values_respect_the_conservative_lower_bound(artifact):
    """The +1 correction means p can never be 0: the floor is 1/(n+1)."""
    n = artifact.calibrator.n
    extreme = np.array([1e9, -1e9])
    p = artifact.calibrator.p_values(extreme)
    assert p[0] == pytest.approx(1.0 / (n + 1))
    assert p[1] == pytest.approx(1.0)
    assert p.min() > 0.0


def test_p_value_decreases_as_score_increases(artifact):
    """Direction check: LOW p means anomalous."""
    scores = np.linspace(-1.0, 1.0, 50)
    p = artifact.calibrator.p_values(scores)
    assert np.all(np.diff(p) <= 1e-12), "p-values must be non-increasing in score"


def test_realised_alarm_rate_respects_epsilon_on_calibration(splits):
    """The conformal bound, checked on the calibration block itself."""
    train, calibration, _ = splits
    detector = fit_detector(train, columns=FEATURES)
    scores = detector.raw_scores(calibration)
    calibrator = SplitConformalCalibrator(scores, epsilon=0.05)
    assert calibrator.realised_alarm_rate(scores) <= 0.05 + 1e-9


def test_threshold_and_p_value_agree_on_the_decision(artifact, splits):
    """`score >= threshold(eps)` and `conformal_p <= eps` must be the same call."""
    _, _, test = splits
    eps = 0.05
    scores = artifact.detector.raw_scores(test)
    by_threshold = scores >= artifact.threshold(eps)
    by_p_value = artifact.calibrator.p_values(scores) <= eps
    np.testing.assert_array_equal(by_threshold, by_p_value)


def test_calibrator_rejects_degenerate_input():
    with pytest.raises(ValueError):
        SplitConformalCalibrator(np.array([]))
    with pytest.raises(ValueError):
        SplitConformalCalibrator(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        SplitConformalCalibrator(np.array([1.0, 2.0])).threshold_at(0.0)


# --- artifact round-trip -------------------------------------------------------


def test_artifact_round_trip_reproduces_identical_scores(artifact, splits, tmp_path):
    """Saving and loading must be lossless to the bit, not merely close."""
    _, _, test = splits
    before = artifact.score_frame(test)

    path = artifact.save(tmp_path / "detector.joblib")
    reloaded = DetectorArtifact.load(path)
    after = reloaded.score_frame(test)

    np.testing.assert_array_equal(before["score"].to_numpy(), after["score"].to_numpy())
    np.testing.assert_array_equal(
        before["conformal_p"].to_numpy(), after["conformal_p"].to_numpy()
    )
    assert reloaded.detector_version == artifact.detector_version
    assert reloaded.feature_columns == artifact.feature_columns
    assert reloaded.threshold() == artifact.threshold()


def test_artifact_writes_an_inspectable_sidecar(artifact, tmp_path):
    import json

    path = artifact.save(tmp_path / "detector.joblib")
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["feature_columns"] == artifact.feature_columns
    assert sidecar["conformal_p_direction"] == "low = anomalous"
    assert sidecar["n_calibration_rows"] == artifact.calibrator.n


# --- feature column order is enforced ------------------------------------------


def test_column_order_is_resolved_by_name_not_position(artifact, splits):
    """A frame whose columns arrive in a different order must score identically,
    because the matrix is built by name.
    """
    _, _, test = splits
    shuffled = test[list(reversed(FEATURES)) + ["window_start", "channel", "label", "split"]]
    np.testing.assert_array_equal(
        artifact.score_frame(test)["score"].to_numpy(),
        artifact.score_frame(shuffled)["score"].to_numpy(),
    )


def test_missing_feature_column_raises(artifact, splits):
    _, _, test = splits
    with pytest.raises(ValueError, match="feature columns missing"):
        artifact.score_frame(test.drop(columns=["beta"]))


def test_missing_feature_column_raises_after_reload(artifact, splits, tmp_path):
    """The enforcement must survive persistence -- that is where it matters."""
    _, _, test = splits
    reloaded = DetectorArtifact.load(artifact.save(tmp_path / "d.joblib"))
    with pytest.raises(ValueError, match="feature columns missing"):
        reloaded.score_frame(test.drop(columns=["gamma"]))


def test_score_features_requires_the_full_vector(artifact):
    """A half-empty feature mapping must raise, not score. A DetectionResult
    built from partial features would carry a number that looks like a score.
    """
    full = {name: 0.1 for name in FEATURES}
    scored = artifact.score_features(full)
    assert set(scored) == {"score", "threshold", "conformal_p", "detector_version"}
    assert 0.0 <= scored["conformal_p"] <= 1.0

    with pytest.raises(ValueError, match="missing features"):
        artifact.score_features({"alpha": 0.1})


def test_score_features_matches_score_frame(artifact):
    """The single-window path used by the service and the batch path used by
    evaluation must not drift apart.
    """
    values = {"alpha": 0.3, "beta": -0.2, "gamma": 1.1}
    single = artifact.score_features(values)
    batch = artifact.score_frame(pd.DataFrame([values])).iloc[0]
    assert single["score"] == pytest.approx(batch["score"])
    assert single["conformal_p"] == pytest.approx(batch["conformal_p"])


# --- library version drift is recorded and surfaced ---------------------------


def test_artifact_records_the_libraries_it_was_fitted_against(artifact):
    """A pickled model is only reproducible against the stack that made it."""
    assert artifact.fitted_with["scikit-learn"]
    assert artifact.fitted_with["numpy"]
    assert set(artifact.fitted_with) >= {"scikit-learn", "numpy", "pyod", "joblib"}


def test_version_drift_is_warned_about_on_load(artifact, tmp_path, caplog):
    """Loading against a different numerical stack must say so. Silence here is
    how scores change without anyone noticing.
    """
    import logging

    import joblib

    path = artifact.save(tmp_path / "d.joblib")
    blob = joblib.load(path)
    blob["fitted_with"] = dict(blob["fitted_with"], **{"scikit-learn": "0.0.1-not-real"})
    joblib.dump(blob, path)

    with caplog.at_level(logging.WARNING, logger="aksha_core.detectors.artifact"):
        reloaded = DetectorArtifact.load(path)
    assert "scikit-learn" in caplog.text
    assert "0.0.1-not-real" in caplog.text
    assert reloaded.detector_version == artifact.detector_version


def test_matching_versions_do_not_warn(artifact, tmp_path, caplog):
    import logging

    path = artifact.save(tmp_path / "d.joblib")
    with caplog.at_level(logging.WARNING, logger="aksha_core.detectors.artifact"):
        DetectorArtifact.load(path)
    assert "not guaranteed to reproduce" not in caplog.text


# --- determinism ---------------------------------------------------------------


def test_fit_is_deterministic_under_a_fixed_seed(splits):
    train, _, test = splits
    a = fit_detector(train, columns=FEATURES)
    b = fit_detector(train, columns=FEATURES)
    np.testing.assert_array_equal(a.raw_scores(test), b.raw_scores(test))
