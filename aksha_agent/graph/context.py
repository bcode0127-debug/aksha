"""Context assembly for `prepare_context`: channel history and k nearest windows.

Reads the committed reference built by `scripts/build_context_reference.py`
from the TRAINING split only, so nothing retrieved here can leak test rows into
an evaluation run's reasoning context.

Everything produced is a summary of named scalars. No raw telemetry array
reaches an LLM (ADR-003) — the nearest-neighbour result is a feature summary of
a comparable window, never its series.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from aksha_agent.graph.schemas import (
    ChannelHistorySummary,
    ChannelStat,
    DetectionSummary,
    NeighbourWindow,
)

DEFAULT_REFERENCE = (
    Path(__file__).resolve().parent.parent.parent
    / "aksha_core"
    / "artifacts"
    / "mission2_context_reference.json"
)

# Ours: how much context the models get. Enough to compare against, small
# enough that the prompt stays cheap — the spike showed thinking tokens already
# dominate the bill, so padding the prompt buys nothing.
TOP_DEVIANT_FEATURES = 6
K_NEAREST = 3


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


class ReferenceContextProvider:
    """Channel history and nearest nominal neighbours from the committed set."""

    def __init__(self, reference_path: str | Path | None = None) -> None:
        self.path = str(reference_path or DEFAULT_REFERENCE)
        data = _load(self.path)
        self.feature_columns: list[str] = data["feature_columns"]
        self.channel_stats: dict = data["channel_stats"]
        self.windows: list[dict] = data["reference_windows"]

    def __call__(
        self, detection: DetectionSummary
    ) -> tuple[ChannelHistorySummary, list[NeighbourWindow]]:
        return self.channel_history(detection), self.nearest(detection)

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
        return ChannelHistorySummary(
            channel_id=detection.channel_id,
            reference_windows=int(stats.get("reference_windows", 0)),
            most_deviant=[stat for _, stat in scored[:TOP_DEVIANT_FEATURES]],
        )

    def nearest(
        self, detection: DetectionSummary, k: int = K_NEAREST
    ) -> list[NeighbourWindow]:
        """k nearest nominal windows on the same channel, by standardised distance.

        Standardised using that channel's own nominal spread, so features
        measured in wildly different units (variance around 1e-8, sample counts
        around 240) contribute comparably instead of the largest-scale feature
        dominating the metric.
        """
        stats = self.channel_stats.get(detection.channel_id)
        if not stats:
            return []

        candidates = [w for w in self.windows if w["channel_id"] == detection.channel_id]
        scored: list[tuple[float, dict]] = []
        for window in candidates:
            total = 0.0
            counted = 0
            for feature in self.feature_columns:
                if feature not in detection.features:
                    continue
                std = stats["std"].get(feature) or 0.0
                if std <= 0.0:
                    continue
                delta = (float(detection.features[feature]) - window["features"][feature]) / std
                total += delta * delta
                counted += 1
            if counted:
                scored.append((math.sqrt(total / counted), window))

        scored.sort(key=lambda pair: pair[0])
        return [
            NeighbourWindow(
                channel_id=window["channel_id"],
                window_start=window["window_start"],
                label=window["label"],
                distance=round(distance, 4),
                # Only the features the model was told about for this window,
                # rounded: full float64 precision is noise in a prompt.
                features={
                    f: round(float(window["features"][f]), 6)
                    for f in self.feature_columns
                    if f in window["features"]
                },
            )
            for distance, window in scored[:k]
        ]
