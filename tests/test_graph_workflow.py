"""Tests for the triage graph.

No LLM calls and no credentials: the agent nodes are swapped for function nodes
returning canned structured output, so the graph's routing, enforcement and
filing logic is exercised for real while the models are not.

The four properties here are the ones whose failure is silent rather than loud:
an unmatched route that ends a branch at exit code 0, a verifier that quietly
sees the confidence it is supposed to be blind to, a severity comparison
inverted on a p-value, and a retry policy that hammers a permanent failure.
"""
from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from aksha_agent.graph import workflow as wf
from aksha_agent.graph.schemas import (
    ChannelHistorySummary,
    DetectionSummary,
    HypothesisKind,
    Severity,
    VerifierInput,
    VerifierStatus,
)

DETECTION = {
    "fragment_id": "frag-test",
    "channel_id": "channel_20",
    "t_start": "2001-12-14T19:00:00Z",
    "t_end": "2001-12-14T20:00:00Z",
    "score": 0.1034,
    "threshold": -0.01182,
    "conformal_p": 0.000626,
    "detector_version": "iforest-conformal-0.1.0",
    "features": {"mean": 0.456, "std": 0.00049},
}


def _context(detection: DetectionSummary):
    return ChannelHistorySummary(channel_id=detection.channel_id, reference_windows=7), []


def run_graph(
    *,
    verifier_status: str,
    hypothesis: str = HypothesisKind.ISOLATED_DEVIATION.value,
    confidence: float = 0.8,
    detection: dict | None = None,
):
    """Run the real graph with the two agent nodes stubbed out.

    Only the agents are replaced. Routing, DEFAULT_ROUTE dispatch, filing and
    severity all execute as they do in production.
    """
    from google.adk import Event, Runner, Workflow
    from google.adk.sessions import InMemorySessionService
    from google.adk.workflow import START

    trace: list[tuple[str, dict]] = []
    real = wf.build_workflow(
        detection=detection or DETECTION,
        incident_id="frag-test",
        trace=lambda node, payload: trace.append((node, payload)),
        context_provider=_context,
        investigate_model="gemini-3.5-flash",
        verify_model="gemini-3.5-flash-lite",
    )

    by_name = {}
    for edge in real.graph.edges:
        for node in (edge.from_node, edge.to_node):
            by_name[node.name] = node

    async def fake_investigate(node_input):
        return Event(
            output={
                "hypothesis": hypothesis,
                "implicated_channel": "channel_20",
                "evidence_refs": ["mean z=+4.1"],
                "confidence": confidence,
            }
        )

    async def fake_verify(node_input):
        # Assert here rather than in a separate test: this is the exact payload
        # the real verifier would receive, built by the real assemble node.
        assert "confidence" not in node_input, node_input
        assert "_investigator_confidence" not in node_input, node_input
        return Event(output={"status": verifier_status, "reason": "stubbed"})

    def rebuilt(node):
        return {"investigate": fake_investigate, "verify": fake_verify}.get(node.name, node)

    edges = []
    for edge in real.graph.edges:
        if edge.from_node.name == "__START__":
            source = START
        else:
            source = rebuilt(edge.from_node)
        edges.append((source, rebuilt(edge.to_node), edge.route))

    # regroup dict edges by source so routing is preserved
    grouped: dict = {}
    order: list = []
    for source, target, route in edges:
        key = id(source)
        if key not in grouped:
            grouped[key] = (source, [], {})
            order.append(key)
        _, plain, routed = grouped[key]
        if route is None:
            plain.append(target)
        else:
            routed[route] = target

    rebuilt_edges = []
    for key in order:
        source, plain, routed = grouped[key]
        for target in plain:
            rebuilt_edges.append((source, target))
        if routed:
            rebuilt_edges.append((source, routed))

    stub_wf = Workflow(name="test_triage", description="stubbed", edges=rebuilt_edges)

    async def _run():
        runner = Runner(
            node=stub_wf,
            app_name="t",
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )
        final = None
        async for event in runner.run_async(user_id="u", session_id="s", new_message=None):
            out = getattr(event, "output", None)
            if isinstance(out, dict) and "routing_destination" in out:
                final = out
        return final

    return asyncio.run(_run()), trace


# --- DEFAULT_ROUTE fires and the incident is still recorded -------------------


def test_default_route_fires_on_unknown_status_and_still_records_incident():
    """The ADR-013 failure mode: an unmatched route ends the branch silently at
    exit code 0. The DEFAULT_ROUTE edge must catch it and still produce an
    incident, flagged.
    """
    incident, trace = run_graph(verifier_status="banana")

    assert incident is not None, "unknown route dropped the incident entirely"
    assert incident["routing_anomaly"] is True
    assert incident["severity"] == Severity.ADVISORY.value
    assert incident["routing_destination"] == "log"

    nodes = [name for name, _ in trace]
    assert "file_report_unroutable" in nodes, nodes
    assert "file_report" not in nodes


def test_known_statuses_do_not_take_the_default_route():
    for status in ("confirm", "reject", "disputed"):
        incident, trace = run_graph(verifier_status=status)
        assert incident is not None, status
        assert incident["routing_anomaly"] is False, status
        assert "file_report_unroutable" not in [n for n, _ in trace], status


def test_graph_declares_a_default_route_on_every_dict_edge():
    """Structural guard: a router added later without a DEFAULT_ROUTE would
    reintroduce the silent-drop failure.
    """
    from google.adk.workflow import DEFAULT_ROUTE

    graph = wf.build_workflow(
        detection=DETECTION,
        incident_id="i",
        investigate_model="gemini-3.5-flash",
        verify_model="gemini-3.5-flash-lite",
    )
    routed_sources = {e.from_node.name for e in graph.graph.edges if e.route is not None}
    default_sources = {
        e.from_node.name for e in graph.graph.edges if e.route == DEFAULT_ROUTE
    }
    assert routed_sources, "no dict edges found — graph shape changed"
    assert routed_sources == default_sources, (
        f"routers without a DEFAULT_ROUTE: {routed_sources - default_sources}"
    )


def test_rejected_and_disputed_never_escalate():
    """ADR-005: disagreement is recorded, never escalated."""
    for status in ("reject", "disputed"):
        incident, _ = run_graph(verifier_status=status)
        assert incident["severity"] == Severity.ADVISORY.value, status
        assert incident["routing_destination"] == "log", status


def test_confirmed_critical_reaches_the_flight_director():
    incident, trace = run_graph(verifier_status="confirm")
    assert incident["severity"] == Severity.CRITICAL.value
    assert incident["routing_destination"] == "flight_director"
    assert "notify_flight_director" in [n for n, _ in trace]


# --- VerifierInput cannot carry confidence (schema level) ---------------------


def _verifier_kwargs():
    return {
        "detection": DetectionSummary(**DETECTION),
        "channel_history": ChannelHistorySummary(
            channel_id="channel_20", reference_windows=7
        ),
        "nearest_labeled": [],
        "hypothesis": HypothesisKind.ISOLATED_DEVIATION,
        "implicated_channel": "channel_20",
        "evidence_refs": [],
    }


def test_verifier_input_has_no_confidence_field():
    assert "confidence" not in VerifierInput.model_fields


def test_verifier_input_rejects_a_smuggled_confidence():
    """ADR-005 is enforced structurally: passing confidence raises rather than
    being silently dropped, so a future caller cannot reintroduce it by accident.
    """
    with pytest.raises(ValidationError):
        VerifierInput(**_verifier_kwargs(), confidence=0.99)


def test_verifier_input_rejects_any_extra_field():
    with pytest.raises(ValidationError):
        VerifierInput(**_verifier_kwargs(), investigator_certainty=0.99)


def test_assemble_node_strips_confidence_before_the_verifier():
    """End-to-end through the real assemble node: the stub verifier asserts the
    absence, so this passing means the drop actually happened in the graph.
    """
    incident, trace = run_graph(verifier_status="confirm", confidence=0.97)
    step = dict(trace)["assemble_verifier_input"]
    assert step["investigator_confidence"] == 0.97
    assert step["confidence_forwarded_to_verifier"] is False
    assert incident is not None


# --- severity direction on the p-value ----------------------------------------


def test_severity_inverts_on_p_value_direction():
    """conformal_p is a p-value: LOW means anomalous, so severity must RISE as p
    FALLS. If the comparison is flipped to `>=`, this fails.
    """
    confirm = VerifierStatus.CONFIRM
    strong = wf.compute_severity(0.0001, confirm, "channel_20")
    medium = wf.compute_severity(0.005, confirm, "channel_20")
    weak = wf.compute_severity(0.9, confirm, "channel_20")

    assert strong == Severity.CRITICAL
    assert medium == Severity.CAUTION
    assert weak == Severity.ADVISORY

    order = {Severity.ADVISORY: 0, Severity.CAUTION: 1, Severity.CRITICAL: 2}
    assert order[strong] > order[medium] > order[weak], (
        "severity must increase as conformal_p decreases; the comparison is inverted"
    )


def test_severity_boundaries_are_inclusive():
    confirm = VerifierStatus.CONFIRM
    assert wf.compute_severity(wf.CRITICAL_P, confirm, "c") == Severity.CRITICAL
    assert wf.compute_severity(wf.CAUTION_P, confirm, "c") == Severity.CAUTION


def test_severity_never_compares_p_against_the_raw_score_threshold():
    """The category error this guards: DetectionResult.threshold is a raw score
    (negative for this detector). Using it as a p-value cut would make every
    incident Advisory while looking like it worked.
    """
    raw_score_threshold = DETECTION["threshold"]
    assert raw_score_threshold < 0
    assert wf.CRITICAL_P > 0 and wf.CAUTION_P > 0
    # a very anomalous p must not collapse to Advisory
    assert (
        wf.compute_severity(0.000626, VerifierStatus.CONFIRM, "channel_20")
        == Severity.CRITICAL
    )


# --- retry allowlist ----------------------------------------------------------


def test_retry_allowlist_excludes_validation_error():
    """ADR-014: a malformed output is deterministic; retrying it burns latency
    and quota on a failure that cannot change.
    """
    assert "ValidationError" not in wf.RETRYABLE_EXCEPTIONS
    assert "ValidationError" not in (wf.AGENT_RETRY.exceptions or [])


def test_retry_allowlist_covers_transient_transport_and_timeout():
    assert "ServerError" in wf.RETRYABLE_EXCEPTIONS
    assert "NodeTimeoutError" in wf.RETRYABLE_EXCEPTIONS


def test_retry_allowlist_excludes_client_error():
    """ClientError conflates 429 with 400 and ADK can only match a class name,
    so admitting it would retry permanent failures.
    """
    assert "ClientError" not in wf.RETRYABLE_EXCEPTIONS


def test_only_agent_nodes_carry_retry_config():
    graph = wf.build_workflow(
        detection=DETECTION,
        incident_id="i",
        investigate_model="gemini-3.5-flash",
        verify_model="gemini-3.5-flash-lite",
    )
    nodes = {}
    for edge in graph.graph.edges:
        for node in (edge.from_node, edge.to_node):
            nodes[node.name] = node

    for name, node in nodes.items():
        retry = getattr(node, "retry_config", None)
        if name in ("investigate", "verify"):
            assert retry is not None, f"{name} should retry transient failures"
            assert retry.exceptions == wf.RETRYABLE_EXCEPTIONS
        else:
            assert retry is None, f"{name} is deterministic and must not retry"


# --- model gate ---------------------------------------------------------------


def test_model_gate_rejects_pre_3_5_and_unversioned_aliases():
    for bad in ("gemini-2.5-flash", "gemini-1.5-pro", "gemini-flash-latest", "gpt-4"):
        with pytest.raises(ValueError):
            wf.require_modern_gemini(bad, "AKSHA_INVESTIGATOR_MODEL")


def test_model_gate_accepts_3_5_and_newer():
    for good in ("gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-4.0-pro"):
        assert wf.require_modern_gemini(good, "AKSHA_INVESTIGATOR_MODEL") == good


def test_the_two_agents_use_different_models():
    """ADR-005: the verifier is a separate call to a different model."""
    graph = wf.build_workflow(
        detection=DETECTION,
        incident_id="i",
        investigate_model="gemini-3.5-flash",
        verify_model="gemini-3.5-flash-lite",
    )
    models = {
        node.name: getattr(node, "model", None)
        for edge in graph.graph.edges
        for node in (edge.from_node, edge.to_node)
        if getattr(node, "model", None)
    }
    assert models["investigate"] != models["verify"]


# --- tracing ------------------------------------------------------------------


def test_every_executed_node_appends_a_trace_step():
    """No node-level callbacks exist in ADK, so each node traces itself. A node
    that forgets is invisible in the audit trail.
    """
    _, trace = run_graph(verifier_status="confirm")
    nodes = [name for name, _ in trace]
    for expected in (
        "prepare_context",
        "assemble_verifier_input",
        "route_by_status",
        "file_report",
        "notify_flight_director",
    ):
        assert expected in nodes, f"{expected} left no trace: {nodes}"


def test_trace_records_zero_llm_calls_for_function_nodes():
    _, trace = run_graph(verifier_status="confirm")
    for name, payload in trace:
        assert payload.get("llm_calls") == 0, f"{name} claimed an LLM call"
