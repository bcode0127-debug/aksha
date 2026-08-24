"""The AKSHA triage graph: an ADK 2 `Workflow` (ADR-013, Pillar 1).

    START -> prepare_context -> investigate -> assemble_explainer_input -> explain
          -> verification_gate -> route_by_status
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
    an unrecognised gate verdict would drop the incident on the floor and
    look like success.

  * There are no node-level callbacks in ADK. Tracing is therefore explicit:
    every node calls the injected `trace` writer itself. Nothing is automatic.

  * `retry_config` exists per node and is opt-in. It is set on the two agent
    nodes only, with an allowlist (ADR-014). Function nodes get none: they are
    deterministic, so a failure there is a bug, not a transient condition.

  * An agent node's structured output reaches the next function node as a plain
    dict, even though the agent's own event carries `output=None`. That is how
    `assemble_explainer_input` receives the investigator's findings.

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
    RecognitionEvidence,
    RoutingDestination,
    Severity,
    ExplainerInput,
    ExplainerOutput,
    Verdict,
)

logger = logging.getLogger(__name__)

# --- model configuration ------------------------------------------------------

INVESTIGATOR_MODEL_ENV = "AKSHA_INVESTIGATOR_MODEL"
EXPLAINER_MODEL_ENV = "AKSHA_EXPLAINER_MODEL"
DEFAULT_INVESTIGATOR_MODEL = "gemini-3.5-flash"
DEFAULT_EXPLAINER_MODEL = "gemini-3.5-flash-lite"

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


def explainer_model() -> str:
    return require_modern_gemini(
        os.environ.get(EXPLAINER_MODEL_ENV, DEFAULT_EXPLAINER_MODEL),
        EXPLAINER_MODEL_ENV,
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

# STEP 4 / determinism. Verified against google-adk 2.3.0: `Agent` exposes
# `generate_content_config`, and google-genai's GenerateContentConfig supports
# temperature and seed. The explainer previously returned 7 rejects and 1 confirm
# across 8 identical inputs; a verdict that varies that much on the same
# evidence is a defect in its own right, independent of which way it leans.
#
# Applied to the EXPLAIN node only, per the packet. The investigator is still
# sampled, so the pipeline is not fully deterministic end to end — its
# hypothesis feeds the explainer, and that residual is reported rather than
# hidden.
def _deterministic_config():
    from google.genai import types

    return types.GenerateContentConfig(temperature=0.0, seed=20260822)



# --- deterministic verification gate ------------------------------------------

# OURS. The gate decides, alone. The model never touches the verdict.
#
# Measured on 200 held-out train-period faults, given the SAME distance the
# model was handed: the threshold confirmed 177/200 (88.5%) of detector-flagged
# faults, the LLM 103/200 (51.5%) -- re-measured live 2026-08-23, post-#16
# (scripts/eval_triage.py --holdout data/processed/mission2_anomaly_holdout.parquet).
# The model reads this signal worse than a comparison against a constant does,
# and the separability analysis says that is a property of the signal (AUC
# 0.610 anomaly vs rare_event, 0.795 anomaly vs nominal+rare_event -- recomputed
# the same session via scripts/calibrate_recognition.py:nearest_rare_distances),
# not of the wording. So the decision is the number's, and the model keeps the
# job it is actually good at: saying why, in language an operator can act on.
#
# An earlier revision let the model escalate a gate-reject to `disputed`. That
# is removed. It made `disputed` mean "the model disagreed", which on the test
# split fired on 33/62 windows and stopped the gate absorbing false alarms —
# and it made the outcome depend on model behaviour that the eight-run
# repeatability check is too small to bound. `disputed` now means one thing
# only, and it is a property of the distance:
#
#   distance >= band.high  -> confirm    outside anything on record
#   distance <= band.low   -> reject     unremarkable
#   otherwise              -> disputed   the calibrated number does not decide
#
# Both bounds come from the calibration artifact. Recalibrating moves them.
def gate_verdict(
    distance: float, band: tuple[float, float]
) -> Verdict:
    """The gate's three-way call. No model input, by construction.

    `band` is (low, high) from the calibration artifact. Note the signature:
    there is nowhere for an LLM verdict to enter, which is why the "the model
    cannot change the outcome" test can be exhaustive rather than a sample.
    """
    low, high = band
    if distance >= high:
        return Verdict.CONFIRM
    if distance <= low:
        return Verdict.REJECT
    return Verdict.DISPUTED


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
    final_verdict: Verdict | None,
    channel_id: str,
    criticality: dict[str, str] | None = None,
) -> Severity:
    """Deterministic severity. No model call, no randomness.

    A rejected or disputed verdict never escalates (ADR-005). `disputed` now
    means the calibrated distance fell inside the band and did not decide —
    which is a reason to log and let a human look, not a reason to wake one.
    """
    if final_verdict in (Verdict.REJECT, Verdict.DISPUTED):
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


def _as_status(value) -> Verdict | None:
    return Verdict(value) if value in {s.value for s in Verdict} else None


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

EXPLAINER_INSTRUCTION = """You are explaining one flagged telemetry window to a spacecraft operator.

A deterministic gate has ALREADY decided whether this window is a fault, by
comparing a calibrated distance against fixed bounds. That decision is final and
nothing you write can change it. You are not being asked to check it, approve it
or overturn it.

Your job is the explanation: say what the evidence actually shows, in language
an operator can act on.

WHAT THE NUMBER MEANS, precisely

`recognition.distance_to_nearest_expected` is how far this window sits from the
nearest previously-labelled rare event on this channel, in standardised units.

`recognition.percentile_among_rare_events` places that distance in the spread of
distances that labelled rare events themselves exhibit.

Read this as OUTLIERNESS, not as resemblance. A small distance does NOT mean the
window is a known rare event — routine nominal windows are on average the
closest of all to rare-event exemplars, because they are unremarkable in every
direction. What the distance tracks is how far out on the tail a window sits:
larger means further from anything previously recorded, and that is what
correlates with a genuine fault.

YOUR OUTPUT

`reason` is the field that matters. One or two sentences, citing the distance,
the percentile, and the feature z-scores that carry the story. This is the text
an operator reads in the alert, so write it for someone deciding whether to act.

`status` is your own independent read, recorded alongside the gate's verdict so
that disagreement between the two can be measured afterwards. It is an audit
record, not a vote:
- confirm  : the evidence looks like a genuine fault
- reject   : the evidence looks like routine or expected behaviour
- disputed : the evidence genuinely does not settle it

Because your status changes nothing, there is no reason to hedge it or to guess
what the gate concluded. State what you actually see; a disagreement that is
recorded honestly is more useful than an agreement that is manufactured.

conformal_p is a calibrated p-value: LOW means unlike nominal data. Everything
you are shown was already flagged as deviant, so deviance alone distinguishes
nothing here."""


# --- the graph ----------------------------------------------------------------


def build_workflow(
    detection: dict,
    incident_id: str,
    trace: TraceWriter | None = None,
    context_provider: ContextProvider | None = None,
    investigate_model: str | None = None,
    explain_model: str | None = None,
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

    # The explainer's evidence is one-sided by construction: only the distance to
    # the nearest KNOWN EXPECTED pattern, calibrated. If the provider cannot
    # supply it, the window is treated as unrecognised rather than silently
    # given a comparison it should not have.
    def recognise(det: DetectionSummary) -> RecognitionEvidence:
        getter = getattr(context_provider, "recognition", None)
        if getter is None:
            return RecognitionEvidence(
                distance_to_nearest_expected=9.99e6,
                percentile_among_rare_events=100.0,
                reference_note="No recognition reference available.",
            )
        return getter(det)

    # The investigator's findings travel to file_report through the closure,
    # NOT through the explainer's payload. Putting them in the payload would
    # hand the explainer the confidence ADR-005 forbids it from seeing, and
    # `ExplainerInput` forbids extras, so it would fail input validation anyway.
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
    async def assemble_explainer_input(node_input):
        """Build the explainer's input WITHOUT the investigator's confidence.

        This is where ADR-005's independence stops being a claim and becomes an
        edge in the graph: the confidence is dropped here, and `ExplainerInput`
        forbids extras, so it cannot be reintroduced downstream.
        """
        findings = InvestigateOutput(**node_input)
        history, _ = context_provider(summary)
        explainer_input = ExplainerInput(
            detection=summary,
            channel_history=history,
            recognition=recognise(summary),
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
            "assemble_explainer_input",
            {
                "hypothesis": findings.hypothesis.value,
                "implicated_channel": findings.implicated_channel,
                "evidence_refs": findings.evidence_refs,
                "investigator_confidence": findings.confidence,
                "confidence_forwarded_to_explainer": False,
                "recognition_distance": explainer_input.recognition.distance_to_nearest_expected,
                "recognition_percentile": explainer_input.recognition.percentile_among_rare_events,
                "anomaly_exemplar_forwarded": False,
                "llm_calls": 0,
            },
        )
        return Event(output=explainer_input.model_dump(mode="json"))

    explain = Agent(
        name="explain",
        model=explain_model or explainer_model(),
        mode="single_turn",
        input_schema=ExplainerInput,
        output_schema=ExplainerOutput,
        instruction=EXPLAINER_INSTRUCTION,
        retry_config=AGENT_RETRY,
        timeout=AGENT_TIMEOUT_SECONDS,
        generate_content_config=_deterministic_config(),
    )

    known_statuses = {s.value for s in Verdict}

    # --- function node: the deterministic gate, 0 LLM ---
    def verification_gate(node_input):
        """Decide from the calibrated distance. Record the model's opinion.

        Runs AFTER `explain` so the model's independent read is available to
        write down — but nothing the model returns is consulted in reaching the
        verdict. `gate_verdict()` takes no LLM argument at all.
        """
        raw_status = node_input.get("status")
        llm_status = raw_status if isinstance(raw_status, str) else str(raw_status)
        recognised = llm_status in known_statuses
        llm_verdict = Verdict(llm_status) if recognised else None
        llm_reason = node_input.get("reason") or ""
        if not recognised:
            # Recorded, not routed on. Before the gate existed an unrecognised
            # status was a routing hazard; now it is only an audit fact, but it
            # must still be visible rather than vanishing into a default.
            logger.warning(
                "explainer returned unrecognised status %r; recorded, not applied",
                llm_status,
            )

        evidence = recognise(summary)
        distance = evidence.distance_to_nearest_expected
        band = getattr(context_provider, "ambiguous_band", None)
        threshold = getattr(context_provider, "operating_threshold", None)

        if band is None:
            # No calibrated band: fall back to the two-way cut rather than
            # inventing bounds. Flagged so the incident shows which rule ran.
            if threshold is None:
                raise RuntimeError(
                    "no operating threshold and no ambiguous band in the "
                    "calibration artifact; the gate cannot decide and there is "
                    "no model fallback to defer to"
                )
            band = (threshold, threshold)
            logger.warning("no ambiguous band available; gate using a two-way cut")

        verdict = gate_verdict(distance, band)

        trace(
            "verification_gate",
            {
                "gate_verdict": verdict.value,
                "llm_verdict": llm_verdict.value if llm_verdict else None,
                "final_verdict": verdict.value,
                "gate_distance": distance,
                "gate_threshold": threshold,
                "band_low": band[0],
                "band_high": band[1],
                "gate_llm_agree": llm_verdict is verdict,
                "llm_status_unrecognised": not recognised,
                "llm_reason": llm_reason,
                "llm_calls": 0,
            },
        )
        return Event(
            output={
                # `status` IS the gate's verdict. There is no combining step.
                "status": verdict.value,
                "reason": llm_reason,
                "gate_verdict": verdict.value,
                "llm_verdict": llm_verdict.value if llm_verdict else None,
                "llm_reason": llm_reason,
                "gate_distance": distance,
                "gate_threshold": threshold,
                "band_low": band[0],
                "band_high": band[1],
                "llm_status_unrecognised": not recognised,
            }
        )

    # --- router 1: the gate's verdict ---
    def route_by_status(node_input):
        """Route on the gate's verdict, tolerating one it does not recognise.

        The gate emits a `Verdict` member, so an unrecognised value
        should now be impossible — this used to guard against a model returning
        something off-enum, and the model no longer supplies the routed value
        at all. It is kept because raising here would fail the node, 500 the
        request, and lose the incident on redelivery, which is strictly worse
        than the silent-drop failure DEFAULT_ROUTE exists to prevent (ADR-013).
        The router is the last thing standing between a surprise and a lost
        incident, so it records the surprise and lets the default branch catch
        it.
        """
        raw_status = node_input.get("status")
        status = raw_status if isinstance(raw_status, str) else str(raw_status)
        reason = node_input.get("reason") or ""
        # The explainer returning an off-enum status is an audit fact, not a
        # routing fault — it cannot reach the route any more. Only a bad GATE
        # verdict is a routing anomaly.
        anomalous = status not in known_statuses
        if anomalous:
            logger.warning(
                "gate emitted unrecognised verdict %r; taking DEFAULT_ROUTE", status
            )
        trace(
            "route_by_status",
            {
                "final_verdict": status,
                "llm_reason": reason,
                "route": status,
                "routing_anomaly": anomalous,
                "llm_calls": 0,
            },
        )
        return Event(
            output={
                "final_verdict": status,
                "routing_anomaly": anomalous,
                "gate_verdict": node_input.get("gate_verdict"),
                "llm_verdict": node_input.get("llm_verdict"),
                "llm_reason": node_input.get("llm_reason"),
                "gate_distance": node_input.get("gate_distance"),
                "gate_threshold": node_input.get("gate_threshold"),
                "band_low": node_input.get("band_low"),
                "band_high": node_input.get("band_high"),
            },
            route=status,
        )

    def _file(node_input, *, log_only: bool, node_name: str, emit_route: bool = True):
        status_value = node_input.get("final_verdict")
        status = Verdict(status_value) if status_value in known_statuses else None
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
            gate_verdict=_as_status(node_input.get("gate_verdict")),
            llm_verdict=_as_status(node_input.get("llm_verdict")),
            final_verdict=status,
            llm_reason=node_input.get("llm_reason"),
            gate_distance=node_input.get("gate_distance"),
            gate_threshold=node_input.get("gate_threshold"),
            band_low=node_input.get("band_low"),
            band_high=node_input.get("band_high"),
            severity=severity,
            routing_anomaly=routing_anomaly,
        )
        trace(
            node_name,
            {
                "severity": severity.value,
                "final_verdict": status_value,
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
    # Each gate verdict gets its OWN node rather than sharing one. ADK
    # rejects duplicate edges between the same pair of nodes, so a single
    # shared "log only" target cannot be reached from three route keys; and
    # splitting them is the better shape anyway, because every outcome —
    # including the unroutable one — is then visible as a distinct node in
    # `Workflow.graph.edges`, and so in the generated diagram.
    def file_report(node_input):
        """Confirmed: severity computed from conformal_p, may escalate."""
        return _file(node_input, log_only=False, node_name="file_report")

    def file_report_rejected(node_input):
        """The gate read this as expected operation: recorded, never escalated."""
        return _file(node_input, log_only=True, node_name="file_report_rejected", emit_route=False)

    def file_report_disputed(node_input):
        """The distance fell inside the calibrated band: logged, never escalated."""
        return _file(node_input, log_only=True, node_name="file_report_disputed", emit_route=False)

    def file_report_unroutable(node_input):
        """Reached via DEFAULT_ROUTE — the gate emitted a verdict no branch
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
            (investigate, assemble_explainer_input),
            (assemble_explainer_input, explain),
            (explain, verification_gate),
            (verification_gate, route_by_status),
            # Router 1: the gate's verdict. DEFAULT_ROUTE is load-bearing — without
            # it an unrecognised status ends the branch silently at exit code 0
            # (ADR-013). Every key needs a distinct target: ADK rejects
            # duplicate edges between the same node pair.
            (
                route_by_status,
                {
                    Verdict.CONFIRM.value: file_report,
                    Verdict.REJECT.value: file_report_rejected,
                    Verdict.DISPUTED.value: file_report_disputed,
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
