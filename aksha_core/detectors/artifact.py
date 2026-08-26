"""Persisted detector: model, scaler, calibration, column order, version.

Pure numpy/pandas/scikit-learn/PyOD/joblib. No `google.*`, ADK or vertexai
imports (ADR-002).

The artifact is committed to the repository on purpose (see `.gitignore`, which
deliberately does not exclude `aksha_core/artifacts/`) so that a clone can score
a window without first obtaining the 3.8 GB Mission2 download and retraining.

`feature_columns` is stored as fitted state and enforced on every scoring call.
Feature order is exactly the kind of thing that fails silently: a matrix built
in the wrong order still has the right shape and still produces numbers, they
are just meaningless. Enforcing it turns that into an exception.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aksha_core.conformal.split import DEFAULT_EPSILON, SplitConformalCalibrator
from aksha_core.detectors.iforest import FittedDetector

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DEFAULT_ARTIFACT_NAME = "mission2_iforest.joblib"

# Libraries whose version can change a pickled model's behaviour. scikit-learn
# warns loudly when unpickling across versions ("may lead to breaking code or
# invalid results"); numpy and pyod can change results without warning at all.
PINNED_LIBRARIES = ("scikit-learn", "numpy", "scipy", "pyod", "joblib")


def library_versions() -> dict[str, str]:
    """Versions of the libraries the fitted model is pickled against."""
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in PINNED_LIBRARIES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover - environment dependent
            out[name] = "unknown"
    return out


@dataclass
class DetectorArtifact:
    """Everything needed to score a window, and nothing that needs refitting."""

    detector_version: str
    detector: FittedDetector
    calibrator: SplitConformalCalibrator
    metadata: dict
    fitted_with: dict[str, str] = field(default_factory=library_versions)

    # --- scoring -------------------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        return self.detector.feature_columns

    def threshold(self, epsilon: float | None = None) -> float:
        return self.calibrator.threshold_at(epsilon)

    def score_frame(
        self, frame: pd.DataFrame, epsilon: float | None = None
    ) -> pd.DataFrame:
        """Score a feature table. Returns score, threshold and conformal_p.

        `conformal_p` is LOW for anomalous windows -- see
        `aksha_core.conformal.split` for the direction and the guarantee.
        """
        epsilon = DEFAULT_EPSILON if epsilon is None else epsilon
        scores = self.detector.raw_scores(frame)
        return pd.DataFrame(
            {
                "score": scores,
                "threshold": self.threshold(epsilon),
                "conformal_p": self.calibrator.p_values(scores),
            },
            index=frame.index,
        )

    def score_features(
        self, features: dict, epsilon: float | None = None
    ) -> dict:
        """Score one window given a plain feature mapping.

        This is the entry point the detector service uses. Missing features
        raise rather than being filled with zeros: a DetectionResult built from
        a half-empty feature vector would carry a number that looks like a
        score and is not one.
        """
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(
                f"missing features required by detector "
                f"{self.detector_version}: {missing}"
            )
        row = pd.DataFrame([{c: features[c] for c in self.feature_columns}])
        scored = self.score_frame(row, epsilon).iloc[0]
        return {
            "score": float(scored["score"]),
            "threshold": float(scored["threshold"]),
            "conformal_p": float(scored["conformal_p"]),
            "detector_version": self.detector_version,
        }

    # --- persistence ---------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else DEFAULT_ARTIFACT_DIR / DEFAULT_ARTIFACT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "detector_version": self.detector_version,
                "feature_columns": self.detector.feature_columns,
                "impute_values": self.detector.impute_values,
                "scaler": self.detector.scaler,
                "model": self.detector.model,
                "n_train_rows": self.detector.n_train_rows,
                "params": self.detector.params,
                "calibration_scores": self.calibrator.calibration_scores,
                "epsilon": self.calibrator.epsilon,
                "metadata": self.metadata,
                "fitted_with": self.fitted_with,
            },
            path,
            compress=3,
        )
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(self.summary(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> DetectorArtifact:
        path = Path(path) if path else DEFAULT_ARTIFACT_DIR / DEFAULT_ARTIFACT_NAME
        blob = joblib.load(path)
        detector = FittedDetector(
            feature_columns=list(blob["feature_columns"]),
            impute_values=dict(blob["impute_values"]),
            scaler=blob["scaler"],
            model=blob["model"],
            n_train_rows=int(blob.get("n_train_rows", 0)),
            params=dict(blob.get("params", {})),
        )
        calibrator = SplitConformalCalibrator(
            calibration_scores=np.asarray(blob["calibration_scores"]),
            epsilon=float(blob.get("epsilon", DEFAULT_EPSILON)),
        )
        fitted_with = dict(blob.get("fitted_with", {}))
        drifted = {
            name: (fitted, running)
            for name, fitted in fitted_with.items()
            if (running := library_versions().get(name)) and running != fitted
        }
        if drifted:
            # Not fatal: the scores may well be identical. But an unpinned
            # numerical stack behind a pickled model is how they stop being
            # identical without anyone noticing, so it is said out loud.
            logger.warning(
                "detector %s was fitted against %s but is running against %s -- "
                "scores are not guaranteed to reproduce; pin the versions",
                blob["detector_version"],
                {k: v[0] for k, v in drifted.items()},
                {k: v[1] for k, v in drifted.items()},
            )

        return cls(
            detector_version=str(blob["detector_version"]),
            detector=detector,
            calibrator=calibrator,
            metadata=dict(blob.get("metadata", {})),
            fitted_with=fitted_with,
        )

    def summary(self) -> dict:
        """Human-readable sidecar, so the artifact is inspectable without
        unpickling it.
        """
        return {
            "detector_version": self.detector_version,
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "impute_values": self.detector.impute_values,
            "n_train_rows": self.detector.n_train_rows,
            "n_calibration_rows": self.calibrator.n,
            "model_params": self.detector.params,
            "epsilon": self.calibrator.epsilon,
            "threshold_table": {
                str(k): v for k, v in self.calibrator.threshold_table().items()
            },
            "conformal_p_direction": "low = anomalous",
            "fitted_with": self.fitted_with,
            "written_utc": datetime.now(timezone.utc).isoformat(),
            **self.metadata,
        }
