"""Detector service — fast path.

Receives a Pub/Sub push envelope on `telemetry-in`, emits a stub
DetectionResult (no real detection yet), writes it to Firestore, and
republishes it to `triage`. See TRD section 1 and 5.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os

from fastapi import FastAPI, Request, Response
from google.cloud import firestore, pubsub_v1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector_service")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
DETECTOR_VERSION = "stub-0.0.1"
TRIAGE_TOPIC = "triage"

app = FastAPI()
db = firestore.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()
triage_topic_path = publisher.topic_path(PROJECT_ID, TRIAGE_TOPIC)


def score_stub(fragment: dict) -> dict:
    """No real detection yet — fixed stub scores, proves the pipeline shape."""
    return {
        "fragment_id": fragment["fragment_id"],
        "channel_id": fragment["channel_id"],
        "t_start": fragment["t_start"],
        "t_end": fragment["t_end"],
        "features": fragment.get("features", {}),
        "context": fragment.get("context", {}),
        "score": 0.9,
        "threshold": 0.85,
        "conformal_p": 0.9,
        "detector_version": DETECTOR_VERSION,
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
        detection = score_stub(fragment)
        doc_ref.set({**detection, "published": False})
        logger.info("wrote detection %s (published=False)", idempotency_key)

    future = publisher.publish(triage_topic_path, json.dumps(detection).encode("utf-8"))
    # future.result() blocks; run it off the event loop instead of freezing other requests.
    message_id = await asyncio.get_event_loop().run_in_executor(None, future.result)
    logger.info("published detection %s to %s, message_id=%s", idempotency_key, TRIAGE_TOPIC, message_id)

    doc_ref.update({"published": True})
    logger.info("marked detection %s as published", idempotency_key)

    return Response(status_code=204)
