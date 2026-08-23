"""Triage service — slow path.

Receives a Pub/Sub push envelope on `triage` (a DetectionResult) and runs the
real triage graph: two Gemini agent nodes and seven function nodes
(`aksha_agent.graph.workflow`). Writes the incident to Firestore plus one trace
document per node.

Tracing is explicit because ADK has no node-level callbacks (ADR-013 spike):
each node calls the writer injected here, and the writer appends to
`traces/{incident_id}/steps/{n}`.

The ack deadline on this subscription is 600s (ADR-011). Two LLM calls at the
measured latency sit far inside that, and each agent node carries its own
120s timeout so a hung call fails the node rather than the subscription.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response
from google.cloud import firestore

from aksha_agent.graph.context import ReferenceContextProvider
from aksha_agent.graph.workflow import build_workflow, investigator_model, explainer_model
from aksha_agent.infra.slack import SlackNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("triage_service")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]

app = FastAPI()
db = firestore.Client(project=PROJECT_ID)

# Cold start. Both fail loudly here rather than per-request: a container that
# cannot resolve its models or its context reference should not serve at all.
CONTEXT_PROVIDER = ReferenceContextProvider()
INVESTIGATOR_MODEL = investigator_model()
EXPLAINER_MODEL = explainer_model()

# Webhooks resolve here, once. A missing secret disables that destination and is
# logged; it does not stop the service, because an incident that cannot be
# delivered still needs to be reasoned about and recorded.
NOTIFIER = SlackNotifier(project_id=PROJECT_ID)

logger.info(
    "triage graph ready: investigate=%s explain=%s, context reference %d exemplars "
    "(categories %s), slack destinations %s",
    INVESTIGATOR_MODEL,
    EXPLAINER_MODEL,
    len(CONTEXT_PROVIDER.windows),
    CONTEXT_PROVIDER.categories,
    NOTIFIER.configured or "none",
)


class FirestoreTrace:
    """Appends one trace document per node, in execution order."""

    def __init__(self, client: firestore.Client, incident_id: str) -> None:
        self._steps = client.collection("traces").document(incident_id).collection("steps")
        self._n = 0
        self.records: list[dict] = []

    def __call__(self, node: str, payload: dict) -> None:
        record = {
            "step": self._n,
            "node": node,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._steps.document(str(self._n)).set(record)
        self.records.append(record)
        self._n += 1

    @property
    def llm_calls(self) -> int:
        return sum(int(r.get("llm_calls", 0)) for r in self.records)


@app.post("/")
async def handle_push(request: Request) -> Response:
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService

    envelope = await request.json()
    message = envelope.get("message", {})
    detection = json.loads(base64.b64decode(message.get("data", "")).decode("utf-8"))

    fragment_id = detection["fragment_id"]
    incident_id = fragment_id

    existing = db.collection("incidents").document(incident_id).get()
    if existing.exists and existing.to_dict().get("status") == "closed":
        logger.info("incident %s already closed, skipping", incident_id)
        return Response(status_code=204)

    # The graph consumes DetectionSummary's fields; the detector adds a few of
    # its own (is_anomalous, published, context) that the schema forbids.
    allowed = {
        "fragment_id", "channel_id", "t_start", "t_end",
        "score", "threshold", "conformal_p", "detector_version", "features",
    }
    payload = {k: v for k, v in detection.items() if k in allowed}

    trace = FirestoreTrace(db, incident_id)
    workflow = build_workflow(
        detection=payload,
        incident_id=incident_id,
        trace=trace,
        context_provider=CONTEXT_PROVIDER,
        investigate_model=INVESTIGATOR_MODEL,
        explain_model=EXPLAINER_MODEL,
        deliver=NOTIFIER.post,
    )

    runner = Runner(
        node=workflow,
        app_name="aksha_triage",
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    started = time.perf_counter()
    incident = None
    usage = {"prompt_tokens": 0, "candidate_tokens": 0, "thought_tokens": 0, "total_tokens": 0}

    async for event in runner.run_async(
        user_id="triage-service", session_id=incident_id, new_message=None
    ):
        meta = getattr(event, "usage_metadata", None)
        if meta is not None:
            usage["prompt_tokens"] += meta.prompt_token_count or 0
            usage["candidate_tokens"] += meta.candidates_token_count or 0
            usage["thought_tokens"] += meta.thoughts_token_count or 0
            usage["total_tokens"] += meta.total_token_count or 0
        output = getattr(event, "output", None)
        if isinstance(output, dict) and "routing_destination" in output:
            incident = output

    elapsed = time.perf_counter() - started

    if incident is None:
        # The graph produced no terminal incident. ADR-013 established this is
        # exactly how an unmatched route fails — silently, with a clean exit —
        # so it is surfaced as a genuine failure and left for redelivery.
        logger.error(
            "triage produced no incident for %s after %.1fs (%d trace steps)",
            incident_id, elapsed, len(trace.records),
        )
        return Response(status_code=500)

    incident["status"] = "closed"
    incident["triage_seconds"] = round(elapsed, 2)
    incident["token_usage"] = usage
    incident["llm_calls"] = trace.llm_calls or 2
    incident["models"] = {"investigate": INVESTIGATOR_MODEL, "explain": EXPLAINER_MODEL}

    db.collection("incidents").document(incident_id).set(incident)
    logger.info(
        "incident %s: severity=%s verdict=%s route=%s delivery=%s anomaly_flag=%s "
        "%.1fs tokens=%d (thoughts=%d) steps=%d",
        incident_id,
        incident.get("severity"),
        incident.get("final_verdict"),
        incident.get("routing_destination"),
        incident.get("routing_outcome"),
        incident.get("routing_anomaly"),
        elapsed,
        usage["total_tokens"],
        usage["thought_tokens"],
        len(trace.records),
    )
    return Response(status_code=204)
