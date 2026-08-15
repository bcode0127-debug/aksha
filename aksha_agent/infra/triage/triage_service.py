"""Triage service — slow path.

Receives a Pub/Sub push envelope on `triage` (a DetectionResult) and runs it
through a minimal ADK 2 `Workflow`: two function nodes, zero agents. Purpose
is to prove `google-adk` installs and executes inside Cloud Run without
spending a single LLM call — the agent nodes (investigate, verify) land in a
later packet. Writes an incident doc plus one trace doc to Firestore.
"""
from __future__ import annotations

import base64
import json
import logging
import os

from fastapi import FastAPI, Request, Response
from google.adk import Event, Runner, Workflow
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START
from google.cloud import firestore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("triage_service")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]

app = FastAPI()
db = firestore.Client(project=PROJECT_ID)


def build_workflow(detection: dict) -> Workflow:
    """Builds a fresh two-node Workflow per request, with `detection` captured
    by closure. Each function node's `node_input` is whatever the upstream
    node returned — `assemble` is the graph's entry point, so it ignores its
    own `node_input` and starts the chain from the closed-over detection.
    """

    async def assemble(node_input):
        """Function node, 0 LLM."""
        return Event(output={"detection": detection})

    async def file_report_stub(node_input):
        """Function node, 0 LLM. Deterministic stub incident assembly."""
        d = node_input["detection"]
        return Event(
            output={
                "fragment_id": d["fragment_id"],
                "channel_id": d["channel_id"],
                "detector_version": d["detector_version"],
                "anomaly_score": d["score"],
                "threshold": d["threshold"],
                "conformal_p": d["conformal_p"],
                "severity": "Advisory",
                "status": "open",
            }
        )

    return Workflow(
        name="triage_stub",
        description="Two function nodes, zero agents — proves google-adk executes in Cloud Run.",
        edges=[
            (START, assemble),
            (assemble, file_report_stub),
        ],
    )


@app.post("/")
async def handle_push(request: Request) -> Response:
    envelope = await request.json()
    message = envelope.get("message", {})
    detection = json.loads(base64.b64decode(message.get("data", "")).decode("utf-8"))

    fragment_id = detection["fragment_id"]
    root = build_workflow(detection)

    runner = Runner(
        node=root,
        app_name="triage_stub",
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    incident = None
    async for event in runner.run_async(
        user_id="triage-service",
        session_id=fragment_id,
        new_message=None,
    ):
        if isinstance(event.output, dict) and "status" in event.output:
            incident = event.output

    if incident is None:
        logger.error("workflow produced no incident for fragment %s", fragment_id)
        return Response(status_code=500)

    db.collection("incidents").document(fragment_id).set(incident)
    logger.info("wrote incident %s", fragment_id)

    db.collection("traces").document(fragment_id).collection("steps").document("0").set(
        {"node": "triage_stub", "incident": incident}
    )
    logger.info("wrote trace for %s", fragment_id)

    return Response(status_code=204)
