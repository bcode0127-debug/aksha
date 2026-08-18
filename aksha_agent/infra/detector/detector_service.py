"""Detector service — fast path.

Receives a Pub/Sub push envelope on `telemetry-in`, scores the window with the
committed Isolation Forest + split conformal artifact, writes the
DetectionResult to Firestore, and republishes it to `triage`. See TRD sections
1 and 5.

The artifact is loaded once at cold start, not per request: it is ~1.8 MB and
unpickling it on every push would dominate the fast path's latency budget.

`conformal_p` is a conformal p-value — LOW means anomalous. See
`aksha_core.conformal.split` for the direction and the guarantee. Anything
downstream that reads this field must not treat it as an anomaly confidence.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os

from fastapi import FastAPI, Request, Response
from google.cloud import firestore, pubsub_v1

from aksha_core.detectors.artifact import DetectorArtifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector_service")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
TRIAGE_TOPIC = "triage"

app = FastAPI()
db = firestore.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()
triage_topic_path = publisher.topic_path(PROJECT_ID, TRIAGE_TOPIC)

# Cold-start load. A failure here should kill the container rather than let it
# serve 204s without scoring anything.
ARTIFACT = DetectorArtifact.load()
DETECTOR_VERSION = ARTIFACT.detector_version
logger.info(
    "loaded detector %s: %d features, %d calibration windows, threshold %.6f",
    DETECTOR_VERSION,
    len(ARTIFACT.feature_columns),
    ARTIFACT.calibrator.n,
    ARTIFACT.threshold(),
)


def score_window(fragment: dict) -> dict:
    """Score one window. Raises ValueError if the feature vector is incomplete.

    No defaulting of absent features: a DetectionResult assembled from a
    partial vector would carry a number that looks like a score and is not one.
    """
    scored = ARTIFACT.score_features(fragment.get("features") or {})
    return {
        "fragment_id": fragment["fragment_id"],
        "channel_id": fragment["channel_id"],
        "t_start": fragment["t_start"],
        "t_end": fragment["t_end"],
        "features": fragment.get("features", {}),
        "context": fragment.get("context", {}),
        "score": scored["score"],
        "threshold": scored["threshold"],
        "conformal_p": scored["conformal_p"],
        "is_anomalous": scored["score"] >= scored["threshold"],
        "detector_version": scored["detector_version"],
    }


@app.post("/")
async def handle_push(request: Request) -> Response:
    envelope = await request.json()
    message = envelope.get("message", {})
    fragment = json.loads(base64.b64decode(message.get("data", "")).decode("utf-8"))

    idempotency_key = f"{fragment['fragment_id']}__{DETECTOR_VERSION}"
    doc_ref = db.collection("detections").document(idempotency_key)
    snapshot = doc_ref.get()

    # Idempotency gates on published == true, not document existence. A write
    # can succeed while the publish that follows it fails (crash, quota,
    # permission error); gating on existence would make that failure
    # unrecoverable — redelivery would find the doc, skip, and 204 without
    # ever republishing. Gating on the flag lets redelivery retry the publish.
    if snapshot.exists and snapshot.to_dict().get("published"):
        logger.info("detection %s already published, skipping", idempotency_key)
        return Response(status_code=204)

    if snapshot.exists:
        detection = {k: v for k, v in snapshot.to_dict().items() if k != "published"}
        logger.info("detection %s exists but unpublished, retrying publish", idempotency_key)
    else:
        try:
            detection = score_window(fragment)
        except ValueError as exc:
            # Permanent, not transient: redelivering the same malformed message
            # will fail identically. 400 lets Pub/Sub exhaust its attempts and
            # dead-letter it rather than retrying forever.
            logger.error("cannot score %s: %s", idempotency_key, exc)
            return Response(status_code=400)
        doc_ref.set({**detection, "published": False})
        logger.info(
            "wrote detection %s (published=False) score=%.6f conformal_p=%.6g anomalous=%s",
            idempotency_key,
            detection["score"],
            detection["conformal_p"],
            detection["is_anomalous"],
        )

    future = publisher.publish(triage_topic_path, json.dumps(detection).encode("utf-8"))
    # future.result() blocks; run it off the event loop instead of freezing other requests.
    message_id = await asyncio.get_event_loop().run_in_executor(None, future.result)
    logger.info("published detection %s to %s, message_id=%s", idempotency_key, TRIAGE_TOPIC, message_id)

    doc_ref.update({"published": True})
    logger.info("marked detection %s as published", idempotency_key)

    return Response(status_code=204)
