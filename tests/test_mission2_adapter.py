"""Tests for the Mission2 adapter.

These exercise the four properties the adapter's correctness actually rests on,
with synthetic data -- no dataset needed, so they run in CI without the 3.8 GB
Mission2 download.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aksha_core.data import mission2 as m2

NS = 1_000_000_000
W0 = pd.Timestamp("2001-01-01")
W0_NS = W0.value
W1_NS = W0_NS + m2.WINDOW.value


def _dense_hour(step_s: int = 18) -> np.ndarray:
    """Raw timestamps covering the whole window at the nominal cadence."""
    return np.arange(W0_NS, W1_NS, step_s * NS, dtype="int64")


# --- gap features are computed pre-ZOH ----------------------------------------


def test_gap_features_see_holes_that_zoh_would_hide():
    """The point of ordering: gaps must be measured on raw timestamps, because
    after a zero-order hold every window looks perfectly sampled.
    """
    first_quarter = np.arange(W0_NS, W0_NS + 900 * NS, 18 * NS, dtype="int64")
    last_quarter = np.arange(W0_NS + 2700 * NS, W1_NS, 18 * NS, dtype="int64")
    raw = np.concatenate([first_quarter, last_quarter])

    gaps = m2.gap_features(raw, W0_NS, W1_NS)
    assert gaps["gap_fraction"] == pytest.approx(1818 / 3600, abs=1e-3)
    assert gaps["max_gap_seconds"] == pytest.approx(1818.0)
    assert gaps["sample_count"] == raw.size

    # The same window after ZOH is a full, evenly spaced grid: the hole is gone.
    grid = np.arange(W0_NS, W1_NS, m2.RESAMPLE_STEP.value, dtype="int64")
    held = m2.zoh_resample(raw, np.ones(raw.size), grid)
    assert held.size == 200
    assert not np.isnan(held).any()


def test_gap_features_count_window_edges():
    """Silence between the window start and its first sample is a gap too."""
    late_start = np.arange(W0_NS + 1800 * NS, W1_NS, 18 * NS, dtype="int64")
    gaps = m2.gap_features(late_start, W0_NS, W1_NS)
    assert gaps["gap_fraction"] == pytest.approx(0.5, abs=1e-3)


def test_gap_features_on_empty_window():
    gaps = m2.gap_features(np.empty(0, dtype="int64"), W0_NS, W1_NS)
    assert gaps["gap_fraction"] == 1.0
    assert gaps["sample_count"] == 0


# --- a window spanning a large gap is dropped ---------------------------------


def test_window_spanning_large_gap_is_dropped():
    """Two dense hours either side of a fully empty one. The empty hour must not
    survive into the feature table as held values.
    """
    hour = m2.WINDOW.value
    dense_a = np.arange(W0_NS, W0_NS + hour, 18 * NS, dtype="int64")
    dense_c = np.arange(W0_NS + 2 * hour, W0_NS + 3 * hour, 18 * NS, dtype="int64")
    stamps = np.concatenate([dense_a, dense_c])

    series = pd.Series(
        np.arange(stamps.size, dtype="float64"),
        index=pd.to_datetime(stamps),
    )
    grid = m2.window_grid(series.index.min(), series.index.max())
    frame = m2.channel_windows(series, grid)

    kept = set(frame["window_start"])
    assert W0 in kept
    assert W0 + pd.Timedelta(hours=2) in kept
    assert W0 + pd.Timedelta(hours=1) not in kept, "empty hour should be dropped"


def test_window_just_under_threshold_survives():
    """The filter is a threshold, not a ban on any gap at all."""
    gap_s = int(m2.MAX_GAP_FRACTION * 3600) - 60  # comfortably under
    stamps = np.concatenate(
        [
            np.arange(W0_NS, W0_NS + (3600 - gap_s) * NS, 18 * NS, dtype="int64"),
            np.array([W1_NS - NS], dtype="int64"),
        ]
    )
    series = pd.Series(np.ones(stamps.size), index=pd.to_datetime(stamps))
    frame = m2.channel_windows(series, pd.DatetimeIndex([W0]))
    assert frame.shape[0] == 1
    assert frame.iloc[0]["gap_fraction"] <= m2.MAX_GAP_FRACTION


# --- label overlap threshold behaves at the boundary --------------------------


def _one_window_frame() -> pd.DataFrame:
    return pd.DataFrame({"window_start": [W0], "channel": ["channel_18"]})


def _label(start_offset_s: float, duration_s: float, category: str = "Anomaly"):
    start = W0 + pd.Timedelta(seconds=start_offset_s)
    return pd.DataFrame(
        {
            "ID": ["id_1"],
            "Channel": ["channel_18"],
            "StartTime": [start],
            "EndTime": [start + pd.Timedelta(seconds=duration_s)],
            "Category": [category],
        }
    )


def test_label_overlap_just_below_threshold_is_not_applied():
    just_under = m2.MIN_LABEL_OVERLAP * 3600 - 1
    out = m2.assign_labels(_one_window_frame(), _label(0, just_under))
    assert out.iloc[0] == "nominal"


def test_label_overlap_exactly_at_threshold_is_applied():
    exactly = m2.MIN_LABEL_OVERLAP * 3600
    out = m2.assign_labels(_one_window_frame(), _label(0, exactly))
    assert out.iloc[0] == "anomaly"


def test_label_overlap_is_clipped_to_the_window():
    """A label far longer than the window still only counts its intersection --
    and a label that misses the window entirely never applies.
    """
    inside = m2.assign_labels(_one_window_frame(), _label(-7200, 100_000))
    assert inside.iloc[0] == "anomaly"

    outside = m2.assign_labels(_one_window_frame(), _label(7200, 3600))
    assert outside.iloc[0] == "nominal"


def test_anomaly_wins_over_rare_event_on_the_same_window():
    labels = pd.concat(
        [_label(0, 3600, "Rare Event"), _label(0, 3600, "Anomaly")], ignore_index=True
    )
    out = m2.assign_labels(_one_window_frame(), labels)
    assert out.iloc[0] == "anomaly"


def test_labels_do_not_leak_across_channels():
    frame = pd.DataFrame(
        {"window_start": [W0, W0], "channel": ["channel_18", "channel_19"]}
    )
    out = m2.assign_labels(frame, _label(0, 3600))
    assert list(out) == ["anomaly", "nominal"]


# --- no train/calibration window carries a non-nominal label ------------------


def test_splits_land_on_the_documented_boundaries():
    starts = pd.Series(
        [
            m2.TRAIN_END - pd.Timedelta(hours=1),
            m2.TRAIN_END,
            m2.TEST_START - pd.Timedelta(hours=1),
            m2.TEST_START,
        ]
    )
    assert list(m2.assign_splits(starts)) == [
        "train",
        "calibration",
        "calibration",
        "test",
    ]


def test_purge_leaves_train_and_calibration_nominal_only():
    """The guarantee the detector depends on: nothing anomalous is fitted or
    calibrated on. Test keeps every label.
    """
    frame = pd.DataFrame(
        {
            "window_start": [
                m2.TRAIN_END - pd.Timedelta(hours=1),
                m2.TRAIN_END + pd.Timedelta(hours=1),
                m2.TEST_START + pd.Timedelta(hours=1),
                m2.TEST_START + pd.Timedelta(hours=2),
            ],
            "label": ["anomaly", "rare_event", "anomaly", "nominal"],
        }
    )
    frame["split"] = m2.assign_splits(frame["window_start"])

    contaminated = frame["split"].isin(["train", "calibration"]) & (
        frame["label"] != "nominal"
    )
    purged = frame[~contaminated]

    fitted = purged[purged["split"].isin(["train", "calibration"])]
    assert (fitted["label"] == "nominal").all()
    assert set(purged[purged["split"] == "test"]["label"]) == {"anomaly", "nominal"}


# --- degenerate windows must not put NaN into the feature table ---------------


def test_constant_window_yields_finite_features():
    """A channel holding a fixed setpoint for an hour is ordinary telemetry, not
    an error. scipy's skew/kurtosis are NaN there, so the adapter must not pass
    them straight through into the detector's input.
    """
    features = m2.series_features(np.full(200, 3.14))
    assert all(np.isfinite(v) for v in features.values()), features
    assert features["skew"] == 0.0
    assert features["kurtosis"] == 0.0
    assert features["std"] == 0.0
    assert features["slope"] == 0.0


def test_near_constant_window_yields_finite_features():
    x = np.full(200, 3.14)
    x[0] += 1e-15
    assert all(np.isfinite(v) for v in m2.series_features(x).values())


def test_ordinary_window_still_reports_real_moments():
    """The degenerate guard must not flatten genuinely varying windows."""
    rng = np.random.default_rng(0)
    features = m2.series_features(rng.gamma(2.0, size=200))
    assert features["std"] > 0
    assert features["skew"] != 0.0


# --- cross-channel feature ----------------------------------------------------


def test_mahalanobis_is_nan_when_a_channel_is_missing():
    """An incomplete mean vector yields NaN rather than a silently imputed
    distance.
    """
    idx = pd.date_range("2000-01-01", periods=400, freq="h")
    rng = np.random.default_rng(0)
    means = pd.DataFrame(rng.normal(size=(400, 3)), index=idx, columns=list("abc"))
    means.iloc[5, 1] = np.nan

    out = m2.mahalanobis_distances(means, np.asarray(means.index < m2.TRAIN_END))
    assert np.isnan(out.iloc[5])
    assert out.notna().sum() == 399


def test_mahalanobis_flags_an_outlying_window():
    idx = pd.date_range("2000-01-01", periods=400, freq="h")
    rng = np.random.default_rng(0)
    means = pd.DataFrame(rng.normal(size=(400, 3)), index=idx, columns=list("abc"))
    means.iloc[399] = 50.0  # far outside the fitted distribution

    out = m2.mahalanobis_distances(means, np.asarray(means.index < m2.TRAIN_END))
    assert out.iloc[399] > out.iloc[:399].max()


# --- telecommand features -----------------------------------------------------


def test_command_features_count_and_elapsed():
    tc = np.array(
        [W0_NS - 600 * NS, W0_NS + 10 * NS, W0_NS + 20 * NS, W1_NS + NS], dtype="int64"
    )
    out = m2.command_features(tc, pd.DatetimeIndex([W0]))
    assert out.iloc[0]["seconds_since_last_tc"] == pytest.approx(600.0)
    assert out.iloc[0]["tc_count_in_window"] == 2


def test_seconds_since_last_tc_is_nan_before_the_first_command():
    tc = np.array([W1_NS + 600 * NS], dtype="int64")
    out = m2.command_features(tc, pd.DatetimeIndex([W0]))
    assert np.isnan(out.iloc[0]["seconds_since_last_tc"])
    assert out.iloc[0]["tc_count_in_window"] == 0
