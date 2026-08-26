"""Split (inductive) conformal calibration for anomaly scores.

Pure numpy/pandas. No `google.*`, ADK or vertexai imports (ADR-002).

Turns a raw Isolation Forest score -- an uncalibrated number whose scale means
nothing on its own -- into a p-value with a distribution-free guarantee. This is
inductive conformal anomaly detection: fit the model on a proper training set,
score a disjoint calibration set drawn from the same nominal distribution, and
rank each new score against those calibration scores.

    Laxhammar, R. and Falkman, G. (2015). Inductive conformal anomaly detection
    for sequential detection of anomalous sub-trajectories. Annals of
    Mathematics and Artificial Intelligence, 74(1-2), 67-94.
    https://doi.org/10.1007/s10472-013-9381-7

DIRECTION -- read this before using `conformal_p` anywhere.

The nonconformity score is the Isolation Forest decision score, where higher
means more outlying. The p-value runs the other way:

    conformal_p LOW  -> nonconforming -> anomalous
    conformal_p HIGH -> conforms to the calibration distribution -> nominal

This is the standard conformal convention and the opposite of an "anomaly
confidence". A window is flagged when `conformal_p <= epsilon`, never when it
exceeds something.

    p(a) = (|{i : alpha_i >= a}| + 1) / (n + 1)

The +1 in both places is the standard conservative correction; it keeps the
guarantee valid for finite n and bounds p away from zero. Consequently
p is in [1/(n+1), 1] -- with 23,970 calibration windows the smallest attainable
p-value is about 4.2e-5, and no window can ever score p = 0.

THE GUARANTEE: if calibration and test nominal windows are exchangeable, then
for a nominal window P(conformal_p <= epsilon) <= epsilon. Choosing epsilon
therefore chooses a false-alarm rate directly, which is the property the raw
Isolation Forest score does not have. Exchangeability is an assumption, not a
fact, and telemetry drifts: the guarantee holds to the extent the spacecraft's
nominal behaviour after 2001-10-01 resembles its behaviour in the calibration
window. `realised_alarm_rate` exists to check that empirically rather than
trust it.

DEFAULT_EPSILON = 0.01 is ours. It means one false alarm per hundred nominal
windows, which over the 168,410-window test split is on the order of a thousand
alarms -- acceptable for an offline evidence panel, far too many for an operator
console. The operating point is a routing decision, not a detector decision, so
it is stored as a table of thresholds rather than baked in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_EPSILON = 0.01
QUANTILE_GRID: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005, 0.001)


@dataclass
class SplitConformalCalibrator:
    """Calibration scores from a held-out nominal block, sorted ascending."""

    calibration_scores: np.ndarray
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        scores = np.asarray(self.calibration_scores, dtype="float64")
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("calibration_scores must be a non-empty 1-D array")
        if not np.isfinite(scores).all():
            raise ValueError("calibration_scores contains non-finite values")
        self.calibration_scores = np.sort(scores)

    @property
    def n(self) -> int:
        return int(self.calibration_scores.size)

    def p_values(self, scores: np.ndarray) -> np.ndarray:
        """Conformal p-values. Low means anomalous -- see module docstring."""
        scores = np.asarray(scores, dtype="float64")
        # count of calibration scores >= s, via the left insertion point
        at_least_as_extreme = self.n - np.searchsorted(
            self.calibration_scores, scores, side="left"
        )
        return (at_least_as_extreme + 1.0) / (self.n + 1.0)

    def threshold_at(self, epsilon: float | None = None) -> float:
        """Raw-score threshold whose p-value is the largest one still <= epsilon.

        A window is anomalous when its raw score is >= this value, which is the
        same decision as `conformal_p <= epsilon`.
        """
        epsilon = self.epsilon if epsilon is None else epsilon
        if not 0.0 < epsilon <= 1.0:
            raise ValueError(f"epsilon must be in (0, 1], got {epsilon}")

        # the ceil((n+1)(1-eps))-th smallest calibration score, clipped into range
        rank = int(np.ceil((self.n + 1) * (1.0 - epsilon)))
        index = min(max(rank - 1, 0), self.n - 1)
        return float(self.calibration_scores[index])

    def threshold_table(
        self, grid: tuple[float, ...] = QUANTILE_GRID
    ) -> dict[float, float]:
        """Thresholds across a grid of significance levels, so the operating
        point can be chosen downstream without refitting anything.
        """
        return {eps: self.threshold_at(eps) for eps in grid}

    def realised_alarm_rate(self, nominal_scores: np.ndarray, epsilon: float | None = None) -> float:
        """Fraction of supposedly-nominal scores that would alarm at `epsilon`.

        The conformal bound says this should be <= epsilon on exchangeable
        nominal data. Materially above it means the exchangeability assumption
        has broken -- drift between the calibration block and the scored data --
        not that the arithmetic is wrong.
        """
        epsilon = self.epsilon if epsilon is None else epsilon
        p = self.p_values(nominal_scores)
        return float(np.mean(p <= epsilon))
