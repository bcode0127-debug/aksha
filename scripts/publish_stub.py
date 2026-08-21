#!/usr/bin/env python3
"""Publish one telemetry window to `telemetry-in` to exercise the pipeline.

detector-service scores it with the committed IForest + conformal artifact,
writes a DetectionResult to Firestore and republishes to `triage`, where
triage-service runs the ADK workflow and writes an incident + trace.

The two built-in windows are REAL rows from the Mission2 test split, not
invented numbers, so the score the service returns is a real score. Their
offline values (detector iforest-conformal-0.1.0) are:

    nominal     channel_21  score -0.155409  conformal_p 0.443578   -> not flagged
    anomaly     channel_20  score +0.103404  conformal_p 0.000626   -> flagged
    rare_event  channel_23  score +0.098887  conformal_p 0.000751   -> flagged

The full 22-feature vector is required; the service rejects a partial one with
400 rather than scoring a half-empty window.

Usage:
    python3 scripts/publish_stub.py                      # nominal window, fresh id
    python3 scripts/publish_stub.py --window anomaly     # the real anomaly window
    python3 scripts/publish_stub.py --fragment-id frag-x # reuse an id to test idempotency
"""
import argparse
import json
import os
import time

from google.cloud import pubsub_v1

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
TOPIC = "telemetry-in"

WINDOWS = {
    "anomaly": {
        "channel_id": "channel_20",
        "window_start": "2001-12-14T19:00:00Z",
        "window_end": "2001-12-14T20:00:00Z",
        "features": {
            "sample_count": 240.0,
            "total_gap_seconds": 0.0,
            "max_gap_seconds": 18.003,
            "gap_fraction": 0.0,
            "mean": 0.45607159182429313,
            "std": 0.0004937527585627344,
            "var": 2.437917865883099e-07,
            "skew": -0.5206966330871836,
            "kurtosis": 4.42638992701951,
            "min": 0.4540262520313263,
            "max": 0.45820093154907227,
            "mean_abs_change": 0.0003940178521314458,
            "n_peaks": 50.0,
            "smooth_n_peaks": 34.0,
            "diff_peaks": 59.0,
            "diff_var": 3.825080081056955e-07,
            "diff2_peaks": 62.0,
            "diff2_var": 9.50816608389128e-07,
            "slope": 2.741515703497257e-07,
            "mahalanobis": 1.032868566509986,
            "seconds_since_last_tc": 576.332,
            "tc_count_in_window": 53.0
        }
    },
    "nominal": {
        "channel_id": "channel_21",
        "window_start": "2002-12-12T17:00:00Z",
        "window_end": "2002-12-12T18:00:00Z",
        "features": {
            "sample_count": 240.0,
            "total_gap_seconds": 0.0,
            "max_gap_seconds": 18.003,
            "gap_fraction": 0.0,
            "mean": 0.16433769315481186,
            "std": 8.549522651026772e-05,
            "var": 7.309433756041983e-09,
            "skew": 0.5560202714997284,
            "kurtosis": -0.32190550111088045,
            "min": 0.164189413189888,
            "max": 0.164530947804451,
            "mean_abs_change": 1.3848045962539749e-05,
            "n_peaks": 9.0,
            "smooth_n_peaks": 7.0,
            "diff_peaks": 29.0,
            "diff_var": 1.701812026866423e-09,
            "diff2_peaks": 48.0,
            "diff2_var": 3.420814074206244e-09,
            "slope": 6.427528092305246e-07,
            "mahalanobis": 1.4051890013379187,
            "seconds_since_last_tc": 522.864,
            "tc_count_in_window": 67.0
        }
    },
    "rare_event": {
        "channel_id": "channel_23",
        "window_start": "2003-01-26T18:00:00Z",
        "window_end": "2003-01-26T19:00:00Z",
        "features": {
            "sample_count": 240.0,
            "total_gap_seconds": 0.0,
            "max_gap_seconds": 18.003,
            "gap_fraction": 0.0,
            "mean": 0.858165335953,
            "std": 0.018304836934,
            "var": 0.000335067055,
            "skew": -0.218338667857,
            "kurtosis": -1.814408226672,
            "min": 0.834005355835,
            "max": 0.876851081848,
            "mean_abs_change": 0.000236438447,
            "n_peaks": 4.0,
            "smooth_n_peaks": 3.0,
            "diff_peaks": 34.0,
            "diff_var": 8.45854e-07,
            "diff2_peaks": 43.0,
            "diff2_var": 1.777144e-06,
            "slope": 0.000289560951,
            "mahalanobis": 1.779164485358,
            "seconds_since_last_tc": 516.312,
            "tc_count_in_window": 63.0
        }
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", choices=sorted(WINDOWS), default="nominal")
    parser.add_argument("--fragment-id", default=None)
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args()

    sample = WINDOWS[args.window]
    fragment_id = args.fragment_id or f"frag-{args.window}-{int(time.time())}"
    payload = {
        "fragment_id": fragment_id,
        "channel_id": sample["channel_id"],
        "t_start": sample["window_start"],
        "t_end": sample["window_end"],
        "features": sample["features"],
        "context": {"source": "scripts/publish_stub.py", "window": args.window},
    }

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, args.topic)
    message_id = publisher.publish(topic_path, json.dumps(payload).encode("utf-8")).result()
    print(
        f"published {fragment_id} ({args.window}, {len(sample['features'])} features) "
        f"to {args.topic}, message_id={message_id}"
    )


if __name__ == "__main__":
    main()
