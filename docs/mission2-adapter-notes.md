# Mission2 adapter -- notes

What `aksha_core/data/mission2.py` produces, every choice that is ours rather
than the benchmark's, and the numbers from the build. Facts and decisions; no
detector design here.

## Output

`data/processed/mission2_features.parquet` (gitignored, 27.11 MB), one row per
(window, channel): **329,615 rows × 26 columns**.

Rebuild with:

```
python3 -m aksha_core.data.mission2 --stats-out /tmp/m2_stats.json
```

Data location resolves in order: `--data-root`, then `$AKSHA_MISSION2_DIR`,
then `data/esa-adb/mission2/ESA-Mission2` (relative to the repo root, inside
the gitignored `data/` -- populated by `scripts/fetch_mission2.py`). No
absolute path is hardcoded. Before this was fixed the default pointed at
`~/Desktop/aksha-datasets/...`, a path that existed only on the machine that
wrote it -- a fresh clone had no way to reproduce the build.

## Sourced from ESA-ADB, not ours

From `github.com/kplabs-pl/ESA-ADB`, verified against the repo in-session:

| Item | Value | Source |
|---|---|---|
| Resampling grid | 18 s | `notebooks/data-prep/Mission2_semiunsupervised_prep_from_raw.py`, `resampling_rule=pd.Timedelta(seconds=18)` |
| Train cut | 2001-07-01 | `mission2_experiments.py:66`, `validation_splits["21_months"]` |
| Test start | 2001-10-01 | `Mission2_semiunsupervised_prep_from_raw.py:31`, `test_data_split` |
| Lightweight subset | channels 18–28 | `mission2_experiments.py`, `subset_channels` |

Dataset itself: Zenodo record `15237121`, DOI `10.5281/zenodo.15237121`,
CC BY 3.0 IGO. Note the ESA-ADB paper and README both cite the older record
`12528696`; this adapter reads the newer one.

## Ours -- to be re-argued if they become load-bearing

**Window: 1 hour, fixed-time, non-overlapping.** Non-overlapping removes label
leakage between neighbouring windows and guarantees no window straddles a split
boundary. Yields ~13.1k windows per channel in the training block.

**`MAX_GAP_FRACTION = 0.20`.** A window whose raw timestamps leave more than 20%
of its wall-clock hour uncovered is dropped rather than zero-order-held into
existence. **On this data the filter is nearly inert** -- see the drop rate
below -- because the 11 lightweight channels are densely and uniformly sampled.
The mechanism matters more than the value here; the value would need re-arguing
on a sparser channel set.

**`MIN_LABEL_OVERLAP = 0.05`** (180 s of the hour). At 0% a one-second clip at a
window edge flips a whole hour's label, which is noise. Measured effect below:
it costs **no test anomaly windows at all**, and trades ~13% of test rare-event
windows for that label cleanliness.

**`GAP_MULTIPLE = 2.0`.** An inter-sample interval counts as a gap once it
exceeds twice the nominal 18 s cadence -- i.e. once at least one expected sample
is missing.

**Anomaly outranks rare event** when both qualify for the same window.

**Mahalanobis is NaN, not imputed,** for any window where fewer than all 11
channels survived the gap filter. An incomplete mean vector has no honest
distance.

**Degenerate windows report skew/kurtosis as 0.0, not NaN.** A channel holding a
fixed setpoint for an hour is ordinary telemetry; scipy returns NaN there, and
NaN would propagate into the detector. Guarded by a relative-tolerance check
plus a non-finite backstop.

**`seconds_since_last_tc` is NaN before the first telecommand on record** -- "no
command has ever been sent" is not a small elapsed time. 11 rows affected.

## Order of operations

Gap features are computed on **raw timestamps, before** the zero-order hold, and
the drop decision is made there. After a ZOH every window looks perfectly
sampled, so gap information only exists at that point. This ordering is the
subject of a regression test.

1. gap features on raw timestamps → 2. drop if `gap_fraction > 0.20` →
3. ZOH to the 18 s grid (200 points/window) → 4. features on the resampled
series.

## Time convention

`labels.csv` carries ISO 8601 with explicit `Z`; the per-channel pickles carry a
timezone-**naive** `DatetimeIndex`. Everything is normalised to naive UTC so the
two join. Getting this wrong raises `TypeError: Cannot compare tz-naive and
tz-aware timestamps` rather than silently mis-joining.

## Columns

Identity: `window_start`, `channel`, `label`, `split`.

Gap (pre-ZOH): `sample_count`, `total_gap_seconds`, `max_gap_seconds`,
`gap_fraction`.

Series (post-ZOH): `mean`, `std`, `var`, `skew`, `kurtosis`, `min`, `max`,
`mean_abs_change`, `n_peaks`, `smooth_n_peaks`, `diff_peaks`, `diff_var`,
`diff2_peaks`, `diff2_var`, `slope`.

Cross-channel: `mahalanobis` -- distance of the window's 11-channel mean vector,
mean and covariance fitted on the **training block only**, pseudo-inverse so a
near-singular covariance degrades rather than raises.

Command: `seconds_since_last_tc`, `tc_count_in_window` (all 123 command types
pooled; no decay term).

## Build numbers

Span `2000-01-01 00:00:16.284` → `2003-06-30 23:59:52.416`.
Wall 141.9 s, peak RSS 782.6 MB.

### Windows and drop rate

| | |
|---|---|
| Windows considered | 337,128 (30,648 × 11 channels) |
| Kept | 337,106 |
| Dropped by gap filter | **22** (2 per channel, identical across all 11) |
| **Drop rate** | **0.0065%** |

All 11 channels have identical raw row counts (7,387,262) and identical drop
counts -- they share one sampling grid.

Mahalanobis defined for 30,646 / 30,646 windows (100%).

### Rows per split, after purge

| split | nominal | rare_event | anomaly | total |
|---|---|---|---|---|
| train | 137,235 | 0 | 0 | 137,235 |
| calibration | 23,970 | 0 | 0 | 23,970 |
| test | 162,760 | 5,646 | **4** | 168,410 |

7,491 non-nominal windows were purged from train and calibration (3,804 anomaly
+ 3,369 rare_event from train; 318 rare_event from calibration). Test keeps
every label.

### Label overlap threshold: measured effect

Test-split counts as the threshold varies:

| threshold | seconds of the hour | test anomaly | test rare_event |
|---|---|---|---|
| 0% | 0 | 4 | 6,502 |
| 1% | 36 | 4 | 6,359 |
| 2% | 72 | 4 | 6,270 |
| **5% (ours)** | **180** | **4** | **5,646** |
| 10% | 360 | 4 | 4,975 |

Across the whole table 1,372 windows (0.416%) change label between 0% and 5%;
every one of them is a rare_event→nominal transition on train/calibration rows
that are purged anyway, or a rare_event→nominal transition in test. **The
threshold does not change the test anomaly count at any value tried.**

## The number that constrains what can be evaluated

**The test split contains 4 anomaly windows.** They come from a single
annotated anomaly event on 2001-12-14, touching three channels:

| window_start | channel |
|---|---|
| 2001-12-14 19:00:00 | channel_18 |
| 2001-12-14 20:00:00 | channel_18 |
| 2001-12-14 19:00:00 | channel_19 |
| 2001-12-14 19:00:00 | channel_20 |

This is a property of the dataset on this channel subset, not of the adapter --
the underlying `labels.csv` has exactly **one distinct anomaly ID** with a
`StartTime` after 2001-10-01 on channels 18–28, and no choice of window size or
overlap threshold creates more.

Consequences, stated plainly:

- Any anomaly-class metric on test (precision, recall, F-score, MCC restricted
  to anomalies) rests on 4 positives from 1 event. It would not be a
  meaningful number and should not be reported as one.
- The rare-event class is healthy: 5,646 windows over 853 distinct hours across
  all 11 channels. Anything evaluated against `rare_event` has real support.
- ESA-ADB's own paper reaches the same wall on the full Mission2 channel set,
  reporting that no algorithm identified the 9 actual anomalies "in this
  overabundance of rare nominal events", and calling it the mission's main
  challenge (arXiv 2406.17826).

Resolved by ADR-015: the evidence panel evaluates against `rare_event ∪
anomaly`, not against OPSSAT-AD. Mission2 is the build dataset; OPSSAT-AD is
retained only as an operator-requirements citation, not as a source of
headline detection metrics.

Pooling the two categories follows ESA-ADB's own benchmarking convention,
verified in the paper (arXiv 2406.17826): *"There are no algorithms in ESA-ADB
that can explicitly distinguish between anomalies and rare nominal events, so
the results in Table 2 are presented for all events (excluding only
communication gaps)."* Table 2's caption says the same -- *"detection of all
events (excluding communication gaps)"*. Their anomaly-only numbers are
quarantined to Supplementary Table 9, for the same reason we would quarantine
ours: too few positives to carry a headline.
