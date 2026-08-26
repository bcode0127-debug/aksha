"""ESA-ADB Mission2 adapter: raw per-channel series -> labeled feature table.

Turns the 11 "lightweight subset" channels (channel_18 .. channel_28) into one
row per (window, channel), with per-window features, a cross-channel feature, a
telecommand feature, a label, and a split assignment.

Pure numpy/pandas/scipy. No `google.*`, ADK or vertexai imports (ADR-002).

Sourced from ESA-ADB's own repository code
(github.com/kplabs-pl/ESA-ADB), not chosen by us:
  * the 18-second resampling grid
    (`notebooks/data-prep/Mission2_semiunsupervised_prep_from_raw.py`,
    `resampling_rule=pd.Timedelta(seconds=18)`)
  * the split boundaries: train cut at 2001-07-01 (`mission2_experiments.py`,
    `validation_splits["21_months"]`) and test start at 2001-10-01
    (`Mission2_semiunsupervised_prep_from_raw.py`, `test_data_split`)
  * the lightweight channel subset 18-28 (`mission2_experiments.py`,
    `subset_channels`)

OUR CHOICES -- not sourced from the benchmark or any paper, and to be re-argued
if they ever get load-bearing:

  * WINDOW = 1 hour, fixed-time, NON-OVERLAPPING. Non-overlapping is
    deliberate: it removes label leakage between neighbouring windows and
    removes any window straddling a split boundary. ~13k windows per channel
    in the training block is ample for Isolation Forest.

  * MAX_GAP_FRACTION = 0.20. A window whose raw timestamps leave more than 20%
    of its wall-clock hour uncovered by data is dropped rather than
    zero-order-held into existence. Chosen so routine short dropouts survive
    but a window that would be mostly held values never reaches the detector.

  * MIN_LABEL_OVERLAP = 0.05. A labelled interval must cover at least 5% of a
    window (180 s of the hour) for that window to inherit the label. Chosen
    from the label duration distribution on these 11 channels: at 0% a
    one-second clip at a window edge flips a whole hour's label, which is pure
    noise; 5% costs only 5 of 56 anomaly rows structurally (those shorter than
    180 s), the same as a 10% threshold would, while leaving far more rare
    events reachable (18% of labelled intervals are shorter than 5% of a
    window, against 37% shorter than 10%).

  * GAP_MULTIPLE = 2.0. An inter-sample interval counts as a "gap" once it
    exceeds twice the nominal 18 s cadence, i.e. once at least one expected
    sample is missing.

  * Mahalanobis distance is defined only for windows where all 11 channels
    survived the gap filter; otherwise the mean vector is incomplete and the
    feature is NaN rather than imputed.

TIME CONVENTION: `labels.csv` timestamps are ISO 8601 with an explicit `Z`;
the per-channel pickles carry a timezone-naive DatetimeIndex. Everything here
is normalised to timezone-naive UTC so the two join. This is the join
convention recorded in docs/mission2-adapter-notes.md.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

# --- Sourced from ESA-ADB repo code (see module docstring) --------------------
LIGHTWEIGHT_CHANNELS: list[str] = [f"channel_{i}" for i in range(18, 29)]
RESAMPLE_STEP = pd.Timedelta(seconds=18)
TRAIN_END = pd.Timestamp("2001-07-01")
TEST_START = pd.Timestamp("2001-10-01")

# --- Ours (see module docstring) ---------------------------------------------
WINDOW = pd.Timedelta(hours=1)
MAX_GAP_FRACTION = 0.20
MIN_LABEL_OVERLAP = 0.05
GAP_MULTIPLE = 2.0
SMOOTH_WINDOW = 10

ENV_DATA_ROOT = "AKSHA_MISSION2_DIR"
# Inside the repo's gitignored data/, populated by scripts/fetch_mission2.py
# (ADR-015) -- not a path outside the repo, so a fresh clone reproduces the
# build without editing this file. AKSHA_MISSION2_DIR still overrides it.
DEFAULT_DATA_ROOT = "data/esa-adb/mission2/ESA-Mission2"

_NS_PER_S = 1_000_000_000


def data_root(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the Mission2 directory: explicit arg, then env var, then default.

    Never a hardcoded absolute path -- the default is expanded at call time so a
    different machine can point `AKSHA_MISSION2_DIR` somewhere else.
    """
    raw = explicit or os.environ.get(ENV_DATA_ROOT) or DEFAULT_DATA_ROOT
    return Path(raw).expanduser()


# --- Loading ------------------------------------------------------------------


def load_channel(name: str, root: str | os.PathLike | None = None) -> pd.Series:
    """Load one channel's raw series. Lazy by design -- callers iterate channels
    one at a time so the full 100-channel set is never resident.
    """
    path = data_root(root) / "channels" / f"{name}.zip"
    with zipfile.ZipFile(path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as fh:
            frame = pd.read_pickle(io.BytesIO(fh.read()))
    series = frame[frame.columns[0]].astype("float64")
    series.index = pd.to_datetime(series.index)
    return series


def load_telecommand_times(
    root: str | os.PathLike | None = None,
    names: list[str] | None = None,
) -> np.ndarray:
    """All telecommand timestamps, pooled across command types and sorted.

    Pooled because the command features (TRD: time-since-last, count-in-window)
    are about operator activity in general, not about any one command type.
    """
    base = data_root(root)
    meta = pd.read_csv(base / "telecommands.csv")
    names = names if names is not None else meta["Telecommand"].tolist()

    stamps: list[np.ndarray] = []
    for name in names:
        path = base / "telecommands" / f"{name}.zip"
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as z:
            inner = z.namelist()[0]
            with z.open(inner) as fh:
                frame = pd.read_pickle(io.BytesIO(fh.read()))
        stamps.append(pd.to_datetime(frame.index).to_numpy(dtype="datetime64[ns]"))

    if not stamps:
        return np.empty(0, dtype="int64")
    return np.sort(np.concatenate(stamps).astype("int64"))


def load_labels(root: str | os.PathLike | None = None) -> pd.DataFrame:
    """labels.csv joined to anomaly_types.csv on ID, timestamps naive-UTC.

    Returns columns: ID, Channel, StartTime, EndTime, Category.
    """
    base = data_root(root)
    labels = pd.read_csv(base / "labels.csv")
    for col in ("StartTime", "EndTime"):
        labels[col] = pd.to_datetime(labels[col], utc=True).dt.tz_localize(None)
    types = pd.read_csv(base / "anomaly_types.csv")[["ID", "Category"]]
    return labels.merge(types, on="ID", how="left")


def load_channel_meta(root: str | os.PathLike | None = None) -> pd.DataFrame:
    return pd.read_csv(data_root(root) / "channels.csv")


# --- Windowing and per-window features ----------------------------------------


def window_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Non-overlapping window start times covering [start, end), floored to the
    window size so every channel shares one global grid.
    """
    first = start.floor(WINDOW)
    return pd.date_range(first, end, freq=WINDOW, inclusive="left")


def gap_features(times_ns: np.ndarray, w_start_ns: int, w_end_ns: int) -> dict:
    """Gap statistics from RAW timestamps, before any resampling.

    Computed pre-ZOH on purpose: after a zero-order hold every window looks
    perfectly sampled, so gap information only exists at this point.

    The window edges count: the span from the window start to its first sample,
    and from its last sample to the window end, are gaps too.
    """
    duration_s = (w_end_ns - w_start_ns) / _NS_PER_S
    nominal_ns = RESAMPLE_STEP.value
    gap_floor_ns = GAP_MULTIPLE * nominal_ns

    if times_ns.size == 0:
        return {
            "sample_count": 0,
            "total_gap_seconds": duration_s,
            "max_gap_seconds": duration_s,
            "gap_fraction": 1.0,
        }

    edges = np.concatenate(([w_start_ns], times_ns, [w_end_ns]))
    deltas = np.diff(edges)
    gaps = deltas[deltas > gap_floor_ns]

    total_gap_s = float(gaps.sum()) / _NS_PER_S if gaps.size else 0.0
    max_gap_s = float(deltas.max()) / _NS_PER_S
    return {
        "sample_count": int(times_ns.size),
        "total_gap_seconds": total_gap_s,
        "max_gap_seconds": max_gap_s,
        "gap_fraction": total_gap_s / duration_s if duration_s else 1.0,
    }


def zoh_resample(
    times_ns: np.ndarray, values: np.ndarray, grid_ns: np.ndarray
) -> np.ndarray:
    """Zero-order hold onto a fixed grid: each grid point takes the most recent
    sample at or before it.

    Grid points preceding the window's first raw sample have nothing to hold, so
    they take that first sample instead. Only reachable when a window opens
    mid-gap, and such windows are usually dropped by the gap filter first.
    """
    idx = np.searchsorted(times_ns, grid_ns, side="right") - 1
    return values[np.clip(idx, 0, values.size - 1)]


def _peak_count(x: np.ndarray) -> int:
    if x.size < 3:
        return 0
    peaks, _ = signal.find_peaks(x)
    return int(peaks.size)


def series_features(x: np.ndarray) -> dict:
    """Features on the resampled series.

    Skew and kurtosis are undefined for a constant series -- scipy returns NaN
    and warns. A channel holding a fixed setpoint for an hour is ordinary in
    telemetry, so those windows are real data, not errors; they are reported as
    0.0 (no asymmetry, no excess tail) rather than allowed to put NaN into the
    feature table, which would propagate into the detector.
    """
    diff1 = np.diff(x)
    diff2 = np.diff(diff1) if diff1.size > 1 else np.empty(0)

    # Relative tolerance, not == 0: a series that is constant to within floating
    # point still has undefined moments, and scipy returns NaN for it.
    spread = float(np.std(x))
    scale = max(abs(float(np.mean(x))), 1.0)
    degenerate = not np.isfinite(x).all() or spread <= 1e-12 * scale

    if x.size >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        smoothed = np.convolve(x, kernel, mode="valid")
    else:
        smoothed = x

    if x.size >= 2 and not degenerate:
        slope = float(np.polyfit(np.arange(x.size, dtype="float64"), x, 1)[0])
    else:
        slope = 0.0

    skew = 0.0 if degenerate or x.size <= 2 else float(stats.skew(x))
    kurtosis = 0.0 if degenerate or x.size <= 3 else float(stats.kurtosis(x))
    # Backstop: whatever the tolerance above misses must still not reach the
    # feature table as NaN.
    skew = skew if np.isfinite(skew) else 0.0
    kurtosis = kurtosis if np.isfinite(kurtosis) else 0.0

    return {
        "mean": float(np.mean(x)),
        "std": spread,
        "var": float(np.var(x)),
        "skew": skew,
        "kurtosis": kurtosis,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean_abs_change": float(np.mean(np.abs(diff1))) if diff1.size else 0.0,
        "n_peaks": _peak_count(x),
        "smooth_n_peaks": _peak_count(smoothed),
        "diff_peaks": _peak_count(diff1),
        "diff_var": float(np.var(diff1)) if diff1.size else 0.0,
        "diff2_peaks": _peak_count(diff2),
        "diff2_var": float(np.var(diff2)) if diff2.size else 0.0,
        "slope": slope,
    }


def channel_windows(series: pd.Series, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Window one channel: gap features on raw timestamps, drop by gap fraction,
    then zero-order hold and compute series features on what survives.
    """
    times_ns = series.index.to_numpy(dtype="datetime64[ns]").astype("int64")
    values = series.to_numpy(dtype="float64")

    starts_ns = grid.to_numpy(dtype="datetime64[ns]").astype("int64")
    ends_ns = starts_ns + WINDOW.value
    lo = np.searchsorted(times_ns, starts_ns, side="left")
    hi = np.searchsorted(times_ns, ends_ns, side="left")

    step_ns = RESAMPLE_STEP.value
    offsets = np.arange(0, WINDOW.value, step_ns, dtype="int64")

    rows: list[dict] = []
    for i in range(starts_ns.size):
        w_times = times_ns[lo[i] : hi[i]]
        gaps = gap_features(w_times, int(starts_ns[i]), int(ends_ns[i]))
        if gaps["gap_fraction"] > MAX_GAP_FRACTION:
            continue

        resampled = zoh_resample(w_times, values[lo[i] : hi[i]], starts_ns[i] + offsets)
        rows.append(
            {"window_start": grid[i], **gaps, **series_features(resampled)}
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["window_start"])
    return frame


def command_features(tc_times_ns: np.ndarray, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Time since the last telecommand at window start, and telecommand count
    inside the window. No decay term.

    `seconds_since_last_tc` is NaN before the first telecommand on record, not 0
    or a sentinel -- "no command has ever been sent" is not a small elapsed time.
    """
    starts_ns = grid.to_numpy(dtype="datetime64[ns]").astype("int64")
    ends_ns = starts_ns + WINDOW.value

    before = np.searchsorted(tc_times_ns, starts_ns, side="right")
    since = np.full(starts_ns.size, np.nan)
    has_prior = before > 0
    since[has_prior] = (
        starts_ns[has_prior] - tc_times_ns[before[has_prior] - 1]
    ) / _NS_PER_S

    count = np.searchsorted(tc_times_ns, ends_ns, side="left") - np.searchsorted(
        tc_times_ns, starts_ns, side="left"
    )

    return pd.DataFrame(
        {
            "window_start": grid,
            "seconds_since_last_tc": since,
            "tc_count_in_window": count.astype("int64"),
        }
    )


def mahalanobis_distances(means: pd.DataFrame, fit_mask: np.ndarray) -> pd.Series:
    """Mahalanobis distance of each window's 11-channel mean vector, against a
    mean and covariance fitted on `fit_mask` rows only (the training block).

    Uses the pseudo-inverse so a singular or near-singular covariance -- likely
    with tightly coupled channels -- degrades rather than raising.
    """
    matrix = means.to_numpy(dtype="float64")
    complete = ~np.isnan(matrix).any(axis=1)

    fit_rows = matrix[fit_mask & complete]
    if fit_rows.shape[0] <= matrix.shape[1]:
        return pd.Series(np.nan, index=means.index, name="mahalanobis")

    centre = fit_rows.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(fit_rows, rowvar=False))

    out = np.full(matrix.shape[0], np.nan)
    delta = matrix[complete] - centre
    out[complete] = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))
    return pd.Series(out, index=means.index, name="mahalanobis")


# --- Labels and splits ---------------------------------------------------------


def assign_labels(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    min_overlap: float = MIN_LABEL_OVERLAP,
) -> pd.Series:
    """Label each (window, channel) row by time overlap with `labels`.

    A labelled interval applies when it covers at least `min_overlap` of the
    window. Where both categories qualify, anomaly wins over rare_event --
    the more severe call is the safer one to surface.
    """
    category_to_label = {"Anomaly": "anomaly", "Rare Event": "rare_event"}
    out = pd.Series("nominal", index=frame.index, dtype="object")
    if frame.empty:
        return out

    window_ns = WINDOW.value
    min_overlap_ns = min_overlap * window_ns
    starts = frame["window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = starts + window_ns

    for channel, chan_labels in labels.groupby("Channel"):
        rows = np.flatnonzero(frame["channel"].to_numpy() == channel)
        if rows.size == 0:
            continue

        l_start = chan_labels["StartTime"].to_numpy(dtype="datetime64[ns]").astype("int64")
        l_end = chan_labels["EndTime"].to_numpy(dtype="datetime64[ns]").astype("int64")
        categories = chan_labels["Category"].to_numpy()

        for j in range(l_start.size):
            label = category_to_label.get(categories[j])
            if label is None:
                continue
            overlap = np.minimum(ends[rows], l_end[j]) - np.maximum(starts[rows], l_start[j])
            hit = rows[overlap >= min_overlap_ns]
            if hit.size == 0:
                continue
            if label == "anomaly":
                out.iloc[hit] = "anomaly"
            else:
                keep = out.iloc[hit] != "anomaly"
                out.iloc[hit[keep.to_numpy()]] = "rare_event"

    return out


def assign_splits(window_start: pd.Series) -> pd.Series:
    """train <= 2001-07-01 < calibration <= 2001-10-01 < test.

    Windows are non-overlapping and hour-aligned, so no window straddles a
    boundary and the assignment is unambiguous.
    """
    out = pd.Series("test", index=window_start.index, dtype="object")
    out[window_start < TEST_START] = "calibration"
    out[window_start < TRAIN_END] = "train"
    return out


# --- Orchestration -------------------------------------------------------------


def build_feature_table(
    root: str | os.PathLike | None = None,
    channels: list[str] | None = None,
    purge_non_nominal: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the full labeled feature table. Returns (table, stats).

    Channels are loaded and windowed one at a time; only the per-window feature
    frames are retained, never more than one raw series at once.
    """
    base = data_root(root)
    channels = channels if channels is not None else LIGHTWEIGHT_CHANNELS
    stats_out: dict = {"per_channel": {}}

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    per_channel: dict[str, pd.DataFrame] = {}

    for name in channels:
        series = load_channel(name, base)
        spans.append((series.index.min(), series.index.max()))
        grid = window_grid(series.index.min(), series.index.max())
        frame = channel_windows(series, grid)
        frame["channel"] = name
        per_channel[name] = frame
        stats_out["per_channel"][name] = {
            "raw_rows": int(series.size),
            "windows_total": int(grid.size),
            "windows_kept": int(frame.shape[0]),
            "windows_dropped": int(grid.size - frame.shape[0]),
        }
        del series

    table = pd.concat(per_channel.values(), ignore_index=True)

    total_windows = sum(v["windows_total"] for v in stats_out["per_channel"].values())
    total_kept = sum(v["windows_kept"] for v in stats_out["per_channel"].values())
    stats_out["windows_total"] = total_windows
    stats_out["windows_kept"] = total_kept
    stats_out["windows_dropped"] = total_windows - total_kept
    stats_out["drop_rate"] = (
        (total_windows - total_kept) / total_windows if total_windows else 0.0
    )

    # Cross-channel: the 11-channel mean vector per window, covariance from the
    # training block only so no test-period information reaches the fit.
    means = table.pivot_table(
        index="window_start", columns="channel", values="mean", aggfunc="first"
    ).reindex(columns=channels)
    fit_mask = np.asarray(means.index < TRAIN_END)
    maha = mahalanobis_distances(means, fit_mask)
    stats_out["mahalanobis_defined"] = int(maha.notna().sum())
    stats_out["mahalanobis_windows"] = int(maha.size)

    table = table.merge(
        maha.rename("mahalanobis"), left_on="window_start", right_index=True, how="left"
    )

    grid_all = pd.DatetimeIndex(sorted(table["window_start"].unique()))
    table = table.merge(
        command_features(load_telecommand_times(base), grid_all),
        on="window_start",
        how="left",
    )

    labels = load_labels(base)
    labels = labels[labels["Channel"].isin(channels)]
    table["label"] = assign_labels(table, labels)
    table["split"] = assign_splits(table["window_start"])

    stats_out["label_counts_before_purge"] = (
        table.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
    )

    if purge_non_nominal:
        contaminated = table["split"].isin(["train", "calibration"]) & (
            table["label"] != "nominal"
        )
        stats_out["purged_from_train_calibration"] = int(contaminated.sum())
        table = table[~contaminated].reset_index(drop=True)

    stats_out["rows"] = int(table.shape[0])
    stats_out["label_counts"] = (
        table.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
    )
    stats_out["span"] = (min(s for s, _ in spans), max(e for _, e in spans))

    # NaN audit. mahalanobis and seconds_since_last_tc are documented as
    # legitimately NaN in places; anything else showing up here is a defect.
    nan_counts = table.isna().sum()
    stats_out["nan_columns"] = {
        col: int(n) for col, n in nan_counts.items() if n > 0
    }
    return table, stats_out


def _main() -> None:
    import argparse
    import json
    import resource
    import time

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None, help=f"defaults to ${ENV_DATA_ROOT} then {DEFAULT_DATA_ROOT}")
    parser.add_argument("--out", default="data/processed/mission2_features.parquet")
    parser.add_argument("--stats-out", default=None, help="optional path for the stats JSON")
    args = parser.parse_args()

    started = time.perf_counter()
    table, stats = build_feature_table(args.data_root)
    elapsed = time.perf_counter() - started

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats["wall_seconds"] = round(elapsed, 1)
    stats["peak_rss_mb"] = round(peak / 1024 / 1024, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    stats["output"] = str(out_path)
    stats["output_mb"] = round(out_path.stat().st_size / 1024 / 1024, 2)

    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    _main()
