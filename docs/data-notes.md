# OPSSAT-AD data notes

Facts recorded from the pinned Zenodo record and the fetched files, via
`scripts/fetch_data.py`. No design decisions here -- see ADRs for those.

## Pinned record

- Record ID: `15108715`
- DOI: `10.5281/zenodo.15108715`
- Concept DOI (all versions): `10.5281/zenodo.12588358`
- Created: 2025-03-30
- License: CC-BY 4.0 (`cc-by-4.0`)
- Note: the concept DOI previously resolved to record `12588359` (created
  2024-07-09); `15108715` supersedes it with different file contents
  (different checksums and sizes on both CSVs, not just a metadata bump).

## File checksums (MD5, from the Zenodo record)

| File | Size (bytes) | MD5 |
|---|---|---|
| `dataset.csv` | 507,550 | `5246fdc5e4630a4cecbf7fb6bc8b795e` |
| `segments.csv` | 18,987,091 | `72f109630abb933a386106897a631188` |

Both verified against these checksums by `scripts/fetch_data.py` at fetch time.

## `dataset.csv`

- Row count: 2123
- Columns (23 total, file order):

  | # | Column | dtype |
  |---|---|---|
  | 0 | `segment` | int64 |
  | 1 | `anomaly` | int64 |
  | 2 | `train` | int64 |
  | 3 | `channel` | str |
  | 4 | `sampling` | int64 |
  | 5 | `duration` | int64 |
  | 6 | `len` | int64 |
  | 7 | `mean` | float64 |
  | 8 | `var` | float64 |
  | 9 | `std` | float64 |
  | 10 | `kurtosis` | float64 |
  | 11 | `skew` | float64 |
  | 12 | `n_peaks` | int64 |
  | 13 | `smooth10_n_peaks` | int64 |
  | 14 | `smooth20_n_peaks` | int64 |
  | 15 | `diff_peaks` | int64 |
  | 16 | `diff2_peaks` | int64 |
  | 17 | `diff_var` | float64 |
  | 18 | `diff2_var` | float64 |
  | 19 | `gaps_squared` | int64 |
  | 20 | `len_weighted` | int64 |
  | 21 | `var_div_duration` | float64 |
  | 22 | `var_div_len` | float64 |

- Non-feature columns: `segment` (row id), `anomaly` (label), `train` (split), `channel` (channel id) -- 4 columns.
- **Feature columns: 19** (not 18), in file order:
  `sampling`, `duration`, `len`, `mean`, `var`, `std`, `kurtosis`, `skew`, `n_peaks`, `smooth10_n_peaks`, `smooth20_n_peaks`, `diff_peaks`, `diff2_peaks`, `diff_var`, `diff2_var`, `gaps_squared`, `len_weighted`, `var_div_duration`, `var_div_len`

### Split column: `train`

| value | count |
|---|---|
| 0 (test) | 529 |
| 1 (train) | 1594 |

Total: 2123. Test count (529) matches the PRD/TRD's stated 529; train count
(1594) does not match the stated 1494 -- off by 100.

### Label column: `anomaly`

| value | count | proportion |
|---|---|---|
| 0 | 1689 | 79.56% |
| 1 | 434 | 20.44% |

### Channel column: `channel`

9 distinct values:

| channel | count |
|---|---|
| CADC0873 | 593 |
| CADC0872 | 546 |
| CADC0888 | 252 |
| CADC0892 | 211 |
| CADC0874 | 194 |
| CADC0884 | 158 |
| CADC0894 | 144 |
| CADC0890 | 14 |
| CADC0886 | 11 |

### Null counts

All 23 columns: 0 nulls.

## `segments.csv`

- Row count: 303,493
- Columns (8 total, file order): `channel` (str), `timestamp` (str), `value` (float64), `label` (str), `sampling` (int64), `anomaly` (int64), `segment` (int64), `train` (int64)
- `head(3)`:

  | channel | timestamp | value | label | sampling | anomaly | segment | train |
  |---|---|---|---|---|---|---|---|
  | CADC0872 | 2022-06-01T23:42:54.000Z | -0.000021 | anomaly | 1 | 1 | 1 | 1 |
  | CADC0872 | 2022-06-01T23:42:55.000Z | -0.000021 | anomaly | 1 | 1 | 1 | 1 |
  | CADC0872 | 2022-06-01T23:42:56.000Z | -0.000021 | anomaly | 1 | 1 | 1 | 1 |

- Null counts: all 8 columns: 0 nulls.

### Join key to `dataset.csv`

`segment` (int64, present in both files). `set(dataset.csv.segment) ==
set(segments.csv.segment)` is `True` -- both files cover the same 2123
segment IDs (1–2123). Rows per segment in `segments.csv` range from 8 to
1040 (mean ~143).

### Timestamp columns

`segments.csv` has exactly one time-related column: `timestamp`, ISO 8601
UTC with `Z` suffix (e.g. `2022-06-01T23:42:54.000Z`). No second timestamp
column exists in either `dataset.csv` or `segments.csv`. No column
distinguishes onboard vs. ground-receive time anywhere in either file.
