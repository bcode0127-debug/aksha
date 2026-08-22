"""The AKSHA triage graph: an ADK 2 `Workflow` (ADR-013, Pillar 1).

    START -> prepare_context -> investigate -> assemble_verifier_input -> verify
          -> route_by_status
               confirm  -> file_report -> [severity] -> notify_flight_director
                                                      | notify_subsystem_engineer
                                                      | record_to_log
                                                      | record_unroutable (DEFAULT)
               reject   -> file_report_rejected   -> record_to_log
               disputed -> file_report_disputed   -> record_to_log
               DEFAULT  -> file_report_unroutable -> record_to_log

`scripts/dump_graph.py` prints this from `Workflow.graph.edges` rather than
from this comment; if the two ever disagree, the script is right.

Three kinds of work, three homes: predictable work is a function node (zero LLM
calls), a clear rule is an explicit dict-edge router, and only reasoning is an
agent. Two LLM calls per incident, total.

WHAT THE SPIKE ESTABLISHED (ADR-013), and what this file relies on:

  * A router that emits a value matching no dict key does NOT raise. ADK logs a
    line and the branch silently ends, exit code 0, no output. Both routers
    therefore carry a `DEFAULT_ROUTE` edge, and both compute `routing_anomaly`
    themselves by checking the value against their own known set. Without this
    an unrecognised verifier status would drop the incident on the floor and
    look like success.

  * There are no node-level callbacks in ADK. Tracing is therefore explicit:
    every node calls the injected `trace` writer itself. Nothing is automatic.

  * `retry_config` exists per node and is opt-in. It is set on the two agent
    nodes only, with an allowlist (ADR-014). Function nodes get none: they are
    deterministic, so a failure there is a bug, not a transient condition.

  * An agent node's structured output reaches the next function node as a plain
    dict, even though the agent's own event carries `output=None`. That is how
    `assemble_verifier_input` receives the investigator's findings.

MODELS come from the environment and must be Gemini 3.5 or newer (CLAUDE.md).
Verified against Vertex in this project: `gemini-3.5-flash` and
`gemini-3.5-flash-lite` exist only on the **global** endpoint — every regional
endpoint 404s for them — so `GOOGLE_CLOUD_LOCATION` must be `global`.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

from google.adk import Agent, Event, Workflow
from google.adk.workflow import DEFAULT_ROUTE, START, RetryConfig

from aksha_agent.graph.schemas import (
    DetectionSummary,
    HypothesisKind,
    IncidentDoc,
    InvestigateInput,
    InvestigateOutput,
    RoutingDestination,
    Severity,
    VerifierInput,
    VerifierOutput,
    VerifierStatus,
)

logger = logging.getLogger(__name__)

# --- model configuration ------------------------------------------------------

INVESTIGATOR_MODEL_ENV = "AKSHA_INVESTIGATOR_MODEL"
VERIFIER_MODEL_ENV = "AKSHA_VERIFIER_MODEL"
DEFAULT_INVESTIGATOR_MODEL = "gemini-3.5-flash"
DEFAULT_VERIFIER_MODEL = "gemini-3.5-flash-lite"

MIN_GEMINI = (3, 5)
_GEMINI_VERSION = re.compile(r"^gemini-(\d+)\.(\d+)")


def require_modern_gemini(model: str, env_var: str) -> str:
    """Fail fast on a model older than Gemini 3.5, or on an unversioned alias.

    CLAUDE.md forbids the ADK tutorial defaults, which fail Stage One. Aliases
    like `gemini-flash-latest` are rejected too: they carry no version, and the
    spike showed they are AI-Studio-only and 404 on Vertex.
    """
    match = _GEMINI_VERSION.match(model)
    if not match:
        raise ValueError(
            f"{env_var}={model!r} is not a versioned Gemini model id. "
            f"AKSHA requires Gemini {MIN_GEMINI[0]}.{MIN_GEMINI[1]}+ "
            "(CLAUDE.md); unversioned aliases 404 on Vertex."
        )
    version = (int(match.group(1)), int(match.group(2)))
    if version < MIN_GEMINI:
        raise ValueError(
            f"{env_var}={model!r} is older than the required Gemini "
            f"{MIN_GEMINI[0]}.{MIN_GEMINI[1]} (CLAUDE.md)."
        )
    return model


def investigator_model() -> str:
    return require_modern_gemini(
        os.environ.get(INVESTIGATOR_MODEL_ENV, DEFAULT_INVESTIGATOR_MODEL),
        INVESTIGATOR_MODEL_ENV,
    )


def verifier_model() -> str:
    return require_modern_gemini(
        os.environ.get(VERIFIER_MODEL_ENV, DEFAULT_VERIFIER_MODEL),
        VERIFIER_MODEL_ENV,
    )


# --- retry policy (ADR-014) ---------------------------------------------------

# ADK matches this allowlist against bare `type(exception).__name__`, verified
# in-session against google-adk 2.3.0.
#
# `ServerError` is google-genai's 5xx — the spike hit a real 503 UNAVAILABLE on
# its first run, which a plain re-run cleared. `NodeTimeoutError` is documented
# by ADK as retry-compatible.
#
# `ClientError` is deliberately ABSENT. It covers both 429 (transient, worth
# retrying) and 400 (permanent, never worth retrying), and the allowlist can
# only match a class name, so there is no way to admit one without the other.
# Retrying a malformed request five times with backoff to 60s is worse than
# failing fast to the deterministic fallback.
#
# `ValidationError` is absent by ADR-014: a bad output shape is deterministic
# and will fail identically on every attempt.
RETRYABLE_EXCEPTIONS = ["ServerError", "NodeTimeoutError"]

AGENT_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=20.0,
    backoff_factor=2.0,
    exceptions=RETRYABLE_EXCEPTIONS,
)

AGENT_TIMEOUT_SECONDS = 120.0


# --- severity (TRD section 7) -------------------------------------------------

# OURS. These are p-value thresholds, and they are deliberately NOT the
# DetectionResult's `threshold` field.
#
# `threshold` there is a RAW SCORE cut (about -0.0118 for the committed
# detector). `conformal_p` is a probability in [0, 1]. Comparing the two is a
# category error that silently never fires: `conformal_p <= -0.0118` is false
# for every possible p-value, so every incident would come out Advisory and the
# graph would look like it was working.
#
# Direction: conformal_p is a p-value, so LOW means anomalous. Severity rises as
# p falls. `test_severity_inverts_on_p_value_direction` fails if this is flipped.
CRITICAL_P = 0.001
CAUTION_P = 0.01

# OURS, and currently uniform. All 11 lightweight-subset channels sit in
# subsystem_1 and are all flagged Target=YES in channels.csv, so there is no
# real criticality gradient to encode yet. The parameter exists so the mapping
# has somewhere to live when a mission provides one; today it does not
# differentiate, and saying so is better than inventing a per-channel ranking.
CHANNEL_CRITICALITY: dict[str, str] = {}
DEFAULT_CRITICALITY = "standard"


def compute_severity(
    conformal_p: float,
    verifier_status: VerifierStatus | None,
    channel_id: str,
    criticality: dict[str, str] | None = None,
) -> Severity:
    """Deterministic severity. No model call, no randomness.

    A rejected or disputed verdict never escalates (ADR-005): disagreement is a
    first-class outcome that goes to the log tier, not a reason to wake anyone.
    """
    if verifier_status in (VerifierStatus.REJECT, VerifierStatus.DISPUTED):
        return Severity.ADVISORY

    criticality = CHANNEL_CRITICALITY if criticality is None else criticality
    _ = criticality.get(channel_id, DEFAULT_CRITICALITY)  # uniform today

    if conformal_p <= CRITICAL_P:
        return Severity.CRITICAL
    if conformal_p <= CAUTION_P:
        return Severity.CAUTION
    return Severity.ADVISORY


SEVERITY_ROUTE = {
    Severity.CRITICAL.value: RoutingDestination.FLIGHT_DIRECTOR,
    Severity.CAUTION.value: RoutingDestination.SUBSYSTEM_ENGINEER,
    Severity.ADVISORY.value: RoutingDestination.LOG,
}


# --- tracing (STEP 5; no node-level callbacks exist in ADK) -------------------

TraceWriter = Callable[[str, dict], None]

# Delivery is injected rather than imported, so the graph has no hard dependency
# on Slack: offline evaluation and the tests run the real routing nodes with no
# webhook in sight, and `None` means "record only".
DeliveryFn = Callable[[dict], str]


def _null_trace(node: str, payload: dict) -> None:  # pragma: no cover - default
    logger.debug("trace %s: %s", node, payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- context assembly ---------------------------------------------------------

ContextProvider = Callable[[DetectionSummary], tuple[Any, list]]


def _empty_context(detection: DetectionSummary):
    from aksha_agent.graph.schemas import ChannelHistorySummary

    return ChannelHistorySummary(channel_id=detection.channel_id, reference_windows=0), []


# --- instructions -------------------------------------------------------------

INVESTIGATOR_INSTRUCTION = """You are a spacecraft telemetry analyst triaging one flagged window.

You receive:
- the detector's scores for this window
- how the window compares to its own channel's nominal history, as z-scores
- the nearest labelled exemplar of each category on this channel (nominal,
  rare_event, anomaly), each with a standardised distance

conformal_p is a calibrated p-value: LOW means the window is unlike nominal
data. It is NOT a confidence that something is wrong. Plenty of expected
spacecraft behaviour is statistically unusual.

Choose the hypothesis the evidence actually supports, and check the evidence for
each before settling:
- command_induced: telecommand activity coincides with the deviation. Look at
  tc_count_in_window and seconds_since_last_tc against their nominal values.
- correlated_channel: the window moves together with related channels. The
  mahalanobis z-score is the cross-channel signal; a large one points here.
- gap_artifact: missing data explains the shape. Look at gap_fraction and
  max_gap_seconds.
- isolated_deviation: this channel alone departs from its history, and none of
  the above explains it. Do not pick this by default — pick it when you have
  ruled the others out.
- sensor_noise: within plausible noise for this channel.

Cite concrete evidence in evidence_refs using feature names and the numbers you
were actually given. Do not invent features. Set confidence to your own
certainty."""

VERIFIER_INSTRUCTION = """You are an independent verifier adjudicating one flagged telemetry window.

Your question is NOT "does the evidence support the analyst's hypothesis". A
window can be strongly deviant and still be entirely expected behaviour, so that
question has the same answer for faults and for routine rare events, and
answering it would tell the operators nothing.

Your question is: IS THIS A GENUINE FAULT, OR UNUSUAL-BUT-EXPECTED OPERATION?

You receive the window's features, its z-scores against this channel's nominal
history, the analyst's hypothesis, and — most importantly — the nearest
labelled exemplar of each category on this same channel, with a standardised
distance to each:
- a known NOMINAL window
- a known RARE_EVENT window: unusual but expected operation, previously judged
  not to be a fault
- a known ANOMALY window: a genuine fault

Reason explicitly over those three distances. A window that sits closest to the
rare_event exemplar looks like known expected behaviour even when it is far from
nominal. A window closest to the anomaly exemplar looks like a fault. Say which
one it most resembles and by how much.

conformal_p is a calibrated p-value: LOW means unlike nominal. It does not
distinguish a fault from a rare event — both are unlike nominal — so do not
decide on it alone.

Return:
- confirm: a genuine fault; operators should act
- reject: unusual but expected operation; deviant, but not a fault
- disputed: the evidence genuinely does not settle it

Cite the exemplar distances and the feature values that decided it. Rejecting is
a useful, expected outcome, not a failure — most flagged windows on this mission
are rare events rather than faults."""


# --- the graph ----------------------------------------------------------------


def build_workflow(
    detection: dict,
    incident_id: str,
    trace: TraceWriter | None = None,
    context_provider: ContextProvider | None = None,
    investigate_model: str | None = None,
    verify_model: str | None = None,
    deliver: DeliveryFn | None = None,
) -> Workflow:
    """Build a triage workflow for one incident.

    The detection is captured by closure rather than seeded through session
    state: the spike showed `state_delta` is not applied before the first node
    runs, so a node reading it would find nothing.
    """
    trace = trace or _null_trace
    context_provider = context_provider or _empty_context
    summary = DetectionSummary(**detection)

    # The investigator's findings travel to file_report through the closure,
    # NOT through the verifier's payload. Putting them in the payload would
    # hand the verifier the confidence ADR-005 forbids it from seeing, and
    # `VerifierInput` forbids extras, so it would fail input validation anyway.
    # The workflow is built per incident, so this holder is per incident.
    findings_for_incident: dict = {}

    # --- function node: context assembly, 0 LLM ---
    async def prepare_context(node_input):
        history, neighbours = context_provider(summary)
        payload = InvestigateInput(
            detection=summary, channel_history=history, nearest_labeled=neighbours
        )
        trace(
            "prepare_context",
            {
                "reference_windows": history.reference_windows,
                "nearest_labeled": len(neighbours),
                "conformal_p": summary.conformal_p,
                "llm_calls": 0,
            },
        )
        return Event(output=payload.model_dump(mode="json"))

    investigate = Agent(
        name="investigate",
        model=investigate_model or investigator_model(),
        mode="single_turn",
        input_schema=InvestigateInput,
        output_schema=InvestigateOutput,
        instruction=INVESTIGATOR_INSTRUCTION,
        retry_config=AGENT_RETRY,
        timeout=AGENT_TIMEOUT_SECONDS,
    )

    # --- function node: ADR-005 enforcement, 0 LLM ---
    async def assemble_verifier_input(node_input):
        """Build the verifier's input WITHOUT the investigator's confidence.

        This is where ADR-005's independence stops being a claim and becomes an
        edge in the graph: the confidence is dropped here, and `VerifierInput`
        forbids extras, so it cannot be reintroduced downstream.
        """
        findings = InvestigateOutput(**node_input)
        history, neighbours = context_provider(summary)
        verifier_input = VerifierInput(
            detection=summary,
            channel_history=history,
            nearest_labeled=neighbours,
            hypothesis=findings.hypothesis,
            implicated_channel=findings.implicated_channel,
            evidence_refs=findings.evidence_refs,
        )
        findings_for_incident.update(
            hypothesis=findings.hypothesis,
            confidence=findings.confidence,
            implicated_channel=findings.implicated_channel,
            evidence_refs=findings.evidence_refs,
        )
        trace(
            "assemble_verifier_input",
            {
                "hypothesis": findings.hypothesis.value,
                "implicated_channel": findings.implicated_channel,
                "evidence_refs": findings.evidence_refs,
                "investigator_confidence": findings.confidence,
                "confidence_forwarded_to_verifier": False,
                "llm_calls": 0,
            },
        )
        return Event(output=verifier_input.model_dump(mode="json"))

    verify = Agent(
        name="verify",
        model=verify_model or verifier_model(),
        mode="single_turn",
        input_schema=VerifierInput,
        output_schema=VerifierOutput,
        instruction=VERIFIER_INSTRUCTION,
        retry_config=AGENT_RETRY,
        timeout=AGENT_TIMEOUT_SECONDS,
    )

    known_statuses = {s.value for s in VerifierStatus}

    # --- router 1: verifier status ---
    def route_by_status(node_input):
        """Route on the verifier's status, tolerating one it does not recognise.

        Deliberately does NOT parse through `VerifierOutput`. The verify node's
        `output_schema` already constrains the model to the enum, so an unknown
        status should be impossible — but if one ever arrives, raising here
        would fail the node, 500 the request, and lose the incident on
        redelivery. That is strictly worse than the silent-drop failure
        DEFAULT_ROUTE exists to prevent (ADR-013). The router is the last thing
        standing between a surprise and a lost incident, so it records the
        surprise and lets the default branch catch it.
        """
        raw_status = node_input.get("status")
        status = raw_status if isinstance(raw_status, str) else str(raw_status)
        reason = node_input.get("reason") or ""
        anomalous = status not in known_statuses
        if anomalous:
            logger.warning(
                "verifier returned unrecognised status %r; taking DEFAULT_ROUTE", status
            )
        trace(
            "route_by_status",
            {
                "verifier_status": status,
                "verifier_reason": reason,
                "route": status,
                "routing_anomaly": anomalous,
                "llm_calls": 0,
            },
        )
        return Event(
            output={
                "verifier_status": status,
                "verifier_reason": reason,
                "routing_anomaly": anomalous,
            },
            route=status,
        )

    def _file(node_input, *, log_only: bool, node_name: str, emit_route: bool = True):
        status_value = node_input.get("verifier_status")
        status = VerifierStatus(status_value) if status_value in known_statuses else None
        routing_anomaly = bool(node_input.get("routing_anomaly"))

        severity = (
            Severity.ADVISORY
            if log_only
            else compute_severity(summary.conformal_p, status, summary.channel_id)
        )
        incident = IncidentDoc(
            incident_id=incident_id,
            timestamp_utc=_now(),
            channel_id=summary.channel_id,
            fragment_id=summary.fragment_id,
            detector_version=summary.detector_version,
            t_start=summary.t_start,
            t_end=summary.t_end,
            anomaly_score=summary.score,
            threshold=summary.threshold,
            conformal_p=summary.conformal_p,
            features_summary=summary.features,
            investigator_hypothesis=findings_for_incident.get("hypothesis"),
            investigator_confidence=findings_for_incident.get("confidence"),
            implicated_channel=findings_for_incident.get("implicated_channel"),
            evidence_refs=findings_for_incident.get("evidence_refs", []),
            verifier_status=status,
            verifier_reason=node_input.get("verifier_reason"),
            severity=severity,
            routing_anomaly=routing_anomaly,
        )
        trace(
            node_name,
            {
                "severity": severity.value,
                "verifier_status": status_value,
                "log_only": log_only,
                "routing_anomaly": routing_anomaly,
                "conformal_p": summary.conformal_p,
                "llm_calls": 0,
            },
        )
        payload = incident.model_dump(mode="json")
        # Only the node that actually sits behind a dict edge emits a route.
        # Emitting one into a plain edge would leave ADK reporting a route that
        # matched no branch.
        if emit_route:
            return Event(output=payload, route=severity.value)
        return Event(output=payload)

    # --- function nodes: deterministic filing, 0 LLM ---
    #
    # Each verifier outcome gets its OWN node rather than sharing one. ADK
    # rejects duplicate edges between the same pair of nodes, so a single
    # shared "log only" target cannot be reached from three route keys; and
    # splitting them is the better shape anyway, because every outcome —
    # including the unroutable one — is then visible as a distinct node in
    # `Workflow.graph.edges`, and so in the generated diagram.
    def file_report(node_input):
        """Confirmed: severity computed from conformal_p, may escalate."""
        return _file(node_input, log_only=False, node_name="file_report")

    def file_report_rejected(node_input):
        """Verifier rejected the hypothesis: recorded, never escalated."""
        return _file(node_input, log_only=True, node_name="file_report_rejected", emit_route=False)

    def file_report_disputed(node_input):
        """Investigator and verifier disagree (ADR-005): logged, never escalated."""
        return _file(node_input, log_only=True, node_name="file_report_disputed", emit_route=False)

    def file_report_unroutable(node_input):
        """Reached via DEFAULT_ROUTE — the verifier emitted a status no branch
        claims. The incident is still recorded and flagged; the ADR-013 spike
        showed the alternative is a branch that ends silently at exit code 0.
        """
        return _file(node_input, log_only=True, node_name="file_report_unroutable", emit_route=False)

    # --- router 2: severity ---
    def _notify(node_input, destination: RoutingDestination, node_name: str):
        incident = dict(node_input)
        incident["routing_destination"] = destination.value

        # Delivery is attempted, then RECORDED — never assumed. `deliver`
        # returns a status rather than raising, so an outage marks the incident
        # `failed` and keeps it, instead of losing an incident the graph has
        # already spent two LLM calls reasoning about (TRD section 9).
        outcome = deliver(incident) if deliver else "recorded"
        incident["routing_outcome"] = outcome
        incident["routing_timestamp_utc"] = _now()

        trace(
            node_name,
            {
                "routing_destination": destination.value,
                "routing_outcome": outcome,
                "delivered": outcome == "delivered",
                "severity": incident.get("severity"),
                "routing_anomaly": incident.get("routing_anomaly", False),
                "llm_calls": 0,
            },
        )
        return Event(output=incident)

    def notify_flight_director(node_input):
        return _notify(node_input, RoutingDestination.FLIGHT_DIRECTOR, "notify_flight_director")

    def notify_subsystem_engineer(node_input):
        return _notify(
            node_input, RoutingDestination.SUBSYSTEM_ENGINEER, "notify_subsystem_engineer"
        )

    def record_to_log(node_input):
        return _notify(node_input, RoutingDestination.LOG, "record_to_log")

    def record_unroutable(node_input):
        """Severity router's DEFAULT_ROUTE. Same invariant as its sibling above:
        no incident leaves the graph unrecorded.
        """
        incident = dict(node_input)
        incident["routing_anomaly"] = True
        return _notify(incident, RoutingDestination.LOG, "record_unroutable")

    return Workflow(
        name="aksha_triage",
        description="Detect-to-route triage: 2 agent nodes, 7 function nodes, 2 dict-edge routers.",
        edges=[
            (START, prepare_context),
            (prepare_context, investigate),
            (investigate, assemble_verifier_input),
            (assemble_verifier_input, verify),
            (verify, route_by_status),
            # Router 1: verifier status. DEFAULT_ROUTE is load-bearing — without
            # it an unrecognised status ends the branch silently at exit code 0
            # (ADR-013). Every key needs a distinct target: ADK rejects
            # duplicate edges between the same node pair.
            (
                route_by_status,
                {
                    VerifierStatus.CONFIRM.value: file_report,
                    VerifierStatus.REJECT.value: file_report_rejected,
                    VerifierStatus.DISPUTED.value: file_report_disputed,
                    DEFAULT_ROUTE: file_report_unroutable,
                },
            ),
            # Router 2: severity. Only the confirmed path has a severity
            # decision to make; the other three are Advisory by construction and
            # go straight to the log tier on a plain edge.
            (
                file_report,
                {
                    Severity.CRITICAL.value: notify_flight_director,
                    Severity.CAUTION.value: notify_subsystem_engineer,
                    Severity.ADVISORY.value: record_to_log,
                    DEFAULT_ROUTE: record_unroutable,
                },
            ),
            (file_report_rejected, record_to_log),
            (file_report_disputed, record_to_log),
            (file_report_unroutable, record_to_log),
        ],
    )
