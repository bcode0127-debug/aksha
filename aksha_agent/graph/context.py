"""Context assembly for `prepare_context`: channel history and labeled exemplars.

Reads the committed reference built by `scripts/build_context_reference.py`,
which is restricted to the train period (< 2001-07-01), so nothing retrieved
here can leak calibration or test rows into an evaluation run's reasoning
context.

Retrieval is PER CATEGORY, not k-nearest overall. The verifier is asked to
decide whether a window is a genuine fault or unusual-but-expected operation,
and it cannot do that from neighbours that all happen to be nominal — which is
what an overall k-nearest search returns, since nominal windows outnumber the
rest by roughly forty to one. Returning the nearest known nominal, the nearest
known rare_event and the nearest known anomaly, each with its distance, gives
the question something to be answered against.

Everything produced is a summary of named scalars. No raw telemetry array
reaches an LLM (ADR-003) — an exemplar is a feature summary of a comparable
window, never its series.
"""
from __future__ import annotations

import json
import math

import numpy as np
from functools import lru_cache
from pathlib import Path

from aksha_agent.graph.schemas import (
    ChannelHistorySummary,
    ChannelStat,
    DetectionSummary,
    NeighbourWindow,
    RecognitionEvidence,
)

_ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "aksha_core" / "artifacts"
DEFAULT_REFERENCE = _ARTIFACTS / "mission2_context_reference.json"
DEFAULT_CALIBRATION = _ARTIFACTS / "mission2_recognition_calibration.json"

# Ours: how much context the models get. Enough to compare against, small
# enough that the prompt stays cheap — the spike showed thinking tokens already
# dominate the bill, so padding the prompt buys nothing.
TOP_DEVIANT_FEATURES = 6
PER_CATEGORY = 1

# Ours: features whose z-score is ALWAYS reported, whether or not they are among
# the most deviant.
#
# Two of the investigator's five hypotheses can only be chosen on evidence that
# was not reliably reaching it. `command_induced` needs telecommand activity;
# `correlated_channel` needs the cross-channel signal; `gap_artifact` needs the
# sampling gaps. Ranking features purely by |z| meant those three could be
# crowded out by whichever statistical moments happened to be extreme, leaving
# the model able to justify only `isolated_deviation` and `sensor_noise`. This
# is a fix to the input, not to the prompt.
ALWAYS_REPORTED = (
    "mahalanobis",           # cross-channel: correlated_channel
    "seconds_since_last_tc",  # command activity: command_induced
    "tc_count_in_window",     # command activity: command_induced
    "gap_fraction",           # sampling: gap_artifact
)


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


class ReferenceContextProvider:
    """Channel history and nearest nominal neighbours from the committed set."""

    def __init__(
        self,
        reference_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.path = str(reference_path or DEFAULT_REFERENCE)
        self.calibration_path = str(calibration_path or DEFAULT_CALIBRATION)
        self._calibration: dict | None = None
        data = _load(self.path)
        self.feature_columns: list[str] = data["feature_columns"]
        self.channel_stats: dict = data["channel_stats"]
        self.windows: list[dict] = data["reference_windows"]
        self.categories: list[str] = data.get(
            "categories", ["nominal", "rare_event", "anomaly"]
        )
        self.train_cut: str | None = data.get("train_cut")

    def __call__(
        self, detection: DetectionSummary
    ) -> tuple[ChannelHistorySummary, list[NeighbourWindow]]:
        return self.channel_history(detection), self.nearest(detection)

    # --- one-sided recognition evidence (what the verifier decides on) --------

    @property
    def calibration(self) -> dict:
        if self._calibration is None:
            path = Path(self.calibration_path)
            self._calibration = json.loads(path.read_text()) if path.exists() else {}
        return self._calibration

    def _percentile_of(self, distance: float) -> float:
        """Place a distance in the known rare events' own nearest-neighbour
        distribution, by interpolation over the calibrated grid.
        """
        values = self.calibration.get("rare_event_percentile_values")
        grid = self.calibration.get("rare_event_percentile_grid")
        if not values or not grid:
            return float("nan")
        # values is non-decreasing; interp gives the percentile for this distance
        return float(round(float(np.interp(distance, values, grid)), 1))

    def recognition(self, detection: DetectionSummary) -> RecognitionEvidence:
        """The verifier's only comparative evidence: how far this window is from
        the nearest KNOWN expected pattern, calibrated.

        Deliberately one-sided — no anomaly exemplar is computed here, so none
        can reach the verifier even by accident.
        """
        matched = None
        for neighbour in self.nearest(detection):
            if neighbour.label == "rare_event":
                matched = neighbour
                break

        distance = matched.distance if matched else float("inf")
        percentile = self._percentile_of(distance) if matched else 100.0
        reference = self.calibration.get("rare_event_reference", {})
        median = reference.get("p50")
        note = (
            f"Among known expected patterns on this mission, the typical distance to "
            f"the nearest other expected pattern is {median:.2f} "
            f"(p25 {reference.get('p25', float('nan')):.2f}, "
            f"p75 {reference.get('p75', float('nan')):.2f})."
            if median is not None
            else "No calibration available."
        )
        return RecognitionEvidence(
            distance_to_nearest_expected=round(float(distance), 4)
            if matched
            else 9.99e6,
            percentile_among_rare_events=min(max(percentile, 0.0), 100.0)
            if percentile == percentile
            else 100.0,
            matched_exemplar=matched,
            reference_note=note,
        )

    def channel_history(self, detection: DetectionSummary) -> ChannelHistorySummary:
        """The window against its own channel's nominal envelope, as z-scores.

        Only the most deviant features are returned: handing the model all 22
        buries the signal, and the ones nearest their nominal mean say nothing.
        """
        stats = self.channel_stats.get(detection.channel_id)
        if not stats:
            return ChannelHistorySummary(
                channel_id=detection.channel_id, reference_windows=0
            )

        scored: list[tuple[float, ChannelStat]] = []
        for feature in self.feature_columns:
            if feature not in detection.features:
                continue
            mean = stats["mean"].get(feature)
            std = stats["std"].get(feature)
            if mean is None or std is None:
                continue
            value = float(detection.features[feature])
            # A zero-variance feature in training cannot produce a z-score; a
            # departure from it is still notable, so it is reported as such
            # rather than dividing by zero or silently dropping the feature.
            if std <= 0.0:
                z = 0.0 if value == mean else math.inf
            else:
                z = (value - mean) / std
            stat = ChannelStat(
                feature=feature,
                nominal_mean=mean,
                nominal_std=std,
                window_value=value,
                z_score=0.0 if math.isinf(z) else round(z, 4),
            )
            scored.append((abs(z), stat))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = [stat for _, stat in scored[:TOP_DEVIANT_FEATURES]]

        # Guarantee the decision-relevant features are present even when they
        # are not among the most deviant — see ALWAYS_REPORTED.
        already = {stat.feature for stat in chosen}
        by_name = {stat.feature: stat for _, stat in scored}
        for feature in ALWAYS_REPORTED:
            if feature not in already and feature in by_name:
                chosen.append(by_name[feature])

        return ChannelHistorySummary(
            channel_id=detection.channel_id,
            reference_windows=int(stats.get("reference_windows", 0)),
            most_deviant=chosen,
        )

    def _distance(self, detection: DetectionSummary, window: dict, stats: dict) -> float | None:
        """Standardised distance, using the channel's own nominal spread.

        Standardising matters here: features range from variance around 1e-8 to
        sample counts around 240, so an unstandardised metric would be decided
        entirely by whichever feature has the largest units.
        """
        total = 0.0
        counted = 0
        for feature in self.feature_columns:
            if feature not in detection.features or feature not in window["features"]:
                continue
            std = stats["std"].get(feature) or 0.0
            if std <= 0.0:
                continue
            delta = (float(detection.features[feature]) - window["features"][feature]) / std
            total += delta * delta
            counted += 1
        return math.sqrt(total / counted) if counted else None

    def nearest(
        self, detection: DetectionSummary, per_category: int = PER_CATEGORY
    ) -> list[NeighbourWindow]:
        """The nearest labelled exemplar of EACH category, on the same channel.

        Ordered nominal, rare_event, anomaly so the comparison reads
        consistently, and each carries its distance so the model can weigh which
        it actually resembles rather than being told.
        """
        stats = self.channel_stats.get(detection.channel_id)
        if not stats:
            return []

        by_category: dict[str, list[tuple[float, dict]]] = {}
        for window in self.windows:
            if window["channel_id"] != detection.channel_id:
                continue
            distance = self._distance(detection, window, stats)
            if distance is None:
                continue
            by_category.setdefault(window["label"], []).append((distance, window))

        out: list[NeighbourWindow] = []
        for category in self.categories:
            scored = sorted(by_category.get(category, []), key=lambda pair: pair[0])
            for distance, window in scored[:per_category]:
                out.append(
                    NeighbourWindow(
                        channel_id=window["channel_id"],
                        window_start=window["window_start"],
                        label=window["label"],
                        distance=round(distance, 4),
                        # Rounded: full float64 precision is noise in a prompt.
                        features={
                            f: round(float(window["features"][f]), 6)
                            for f in self.feature_columns
                            if f in window["features"]
                        },
                    )
                )
        return out
