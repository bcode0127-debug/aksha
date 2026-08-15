#!/usr/bin/env python3
"""Publish one stub telemetry fragment to telemetry-in.

Used to exercise the walking skeleton end-to-end: detector-service picks
this up, scores it (stub), writes a DetectionResult to Firestore, and
republishes to triage, where triage-service runs the stub Workflow and
writes an incident + trace doc.

Usage:
    python3 scripts/publish_stub.py [fragment_id]

If fragment_id is omitted, a fresh one is generated each run. Pass the same
fragment_id twice to exercise the idempotency check in detector-service.
"""
import json
import os
import sys
import time

from google.cloud import pubsub_v1

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
TOPIC = "telemetry-in"


def main() -> None:
    fragment_id = sys.argv[1] if len(sys.argv) > 1 else f"frag-stub-{int(time.time())}"
    payload = {
        "fragment_id": fragment_id,
        "channel_id": "channel_1",
        "t_start": "2026-08-15T00:00:00Z",
        "t_end": "2026-08-15T00:05:00Z",
        "features": {"mean": 0.5, "std": 0.1, "n_peaks": 3.0},
        "context": {},
    }

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC)
    future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
    message_id = future.result()
    print(f"published {fragment_id} to {TOPIC}, message_id={message_id}")


if __name__ == "__main__":
    main()
