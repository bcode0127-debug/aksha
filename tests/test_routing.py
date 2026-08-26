"""Tests for severity routing and Slack delivery.

No network and no secrets: the notifier's HTTP call and its Secret Manager
client are both injected, so these run in CI.

The property that matters most is the last one. Delivery is best-effort, but
"best-effort" must not quietly become "lost": a webhook failure has to mark the
incident `failed` and still record it, because by that point the graph has
already spent two LLM calls deciding what the incident is.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aksha_agent.graph.schemas import RoutingDestination, Severity, Verdict
from aksha_agent.infra import slack
from tests.test_graph_workflow import DETECTION, run_graph


# --- severity maps to the right destination -----------------------------------


def test_critical_routes_to_the_flight_director():
    incident, trace = run_graph(verifier_status="confirm")  # conformal_p 0.000626
    assert incident["severity"] == Severity.CRITICAL.value
    assert incident["routing_destination"] == RoutingDestination.FLIGHT_DIRECTOR.value
    assert "notify_flight_director" in [n for n, _ in trace]


def test_caution_routes_to_the_subsystem_engineer():
    detection = dict(DETECTION, conformal_p=0.005)  # between CRITICAL_P and CAUTION_P
    incident, trace = run_graph(verifier_status="confirm", detection=detection)
    assert incident["severity"] == Severity.CAUTION.value
    assert incident["routing_destination"] == RoutingDestination.SUBSYSTEM_ENGINEER.value
    assert "notify_subsystem_engineer" in [n for n, _ in trace]


def test_advisory_routes_to_the_log():
    detection = dict(DETECTION, conformal_p=0.9)
    incident, trace = run_graph(verifier_status="confirm", detection=detection)
    assert incident["severity"] == Severity.ADVISORY.value
    assert incident["routing_destination"] == RoutingDestination.LOG.value
    assert "record_to_log" in [n for n, _ in trace]


# --- disputed never escalates (ADR-005, TRD section 7) ------------------------


def test_disputed_never_escalates_even_at_the_most_extreme_p():
    """A disagreement is logged, not escalated -- however anomalous the window
    looks. Escalating on disagreement would page an operator on the strength of
    two models failing to agree.
    """
    detection = dict(DETECTION, conformal_p=1e-9)
    incident, _ = run_graph(verifier_status="disputed", detection=detection)
    assert incident["severity"] == Severity.ADVISORY.value
    assert incident["routing_destination"] == RoutingDestination.LOG.value


def test_rejected_never_escalates():
    detection = dict(DETECTION, conformal_p=1e-9)
    incident, _ = run_graph(verifier_status="reject", detection=detection)
    assert incident["routing_destination"] == RoutingDestination.LOG.value


# --- routing_anomaly goes to log (TRD section 9) ------------------------------


def test_an_unusable_model_status_does_not_disturb_routing():
    """The model's status is no longer routed on, so a garbage value from it is
    an audit fact rather than a routing anomaly. The gate's verdict decides the
    destination exactly as it would have anyway.
    """
    incident, trace = run_graph(verifier_status="not-a-real-status", gate_distance=0.5)
    assert incident["routing_destination"] == RoutingDestination.LOG.value
    assert incident["routing_anomaly"] is False
    assert incident["llm_verdict"] is None
    assert dict(trace)["verification_gate"]["llm_status_unrecognised"] is True


# --- delivery outcome is recorded, and failure does not lose the incident -----


class _FakeNotifier:
    def __init__(self, outcome: str):
        self.outcome = outcome
        self.calls: list[dict] = []

    def post(self, incident: dict) -> str:
        self.calls.append(incident)
        return self.outcome


def test_successful_delivery_is_recorded_as_delivered():
    notifier = _FakeNotifier("delivered")
    incident, trace = run_graph(verifier_status="confirm", deliver=notifier.post)
    assert incident["routing_outcome"] == "delivered"
    assert incident["routing_timestamp_utc"]
    assert len(notifier.calls) == 1
    step = dict(trace)["notify_flight_director"]
    assert step["routing_outcome"] == "delivered"
    assert step["delivered"] is True


def test_webhook_failure_marks_failed_without_losing_the_incident():
    """The whole point of best-effort delivery: the incident survives."""
    notifier = _FakeNotifier("failed")
    incident, trace = run_graph(verifier_status="confirm", deliver=notifier.post)

    assert incident is not None, "a delivery failure lost the incident"
    assert incident["routing_outcome"] == "failed"
    assert incident["severity"] == Severity.CRITICAL.value
    assert incident["routing_destination"] == RoutingDestination.FLIGHT_DIRECTOR.value
    assert incident["routing_timestamp_utc"]
    assert dict(trace)["notify_flight_director"]["delivered"] is False


def test_a_raising_notifier_would_not_be_swallowed_silently():
    """SlackNotifier.post is contracted never to raise. If a future
    implementation breaks that contract, the incident is lost -- so the contract
    is asserted directly on the real class.
    """
    notifier = slack.SlackNotifier(project_id="p", client=object())
    notifier._resolved = True
    notifier._webhooks = {}
    assert notifier.post({"routing_destination": "log"}) == slack.SKIPPED


# --- the notifier itself -------------------------------------------------------


def _notifier_with(url: str = "https://hooks.example/T/B/X") -> slack.SlackNotifier:
    notifier = slack.SlackNotifier(project_id="p", client=object())
    notifier._resolved = True
    notifier._webhooks = {d: url for d in slack.SECRET_IDS}
    return notifier


INCIDENT = {
    "incident_id": "frag-1",
    "channel_id": "channel_20",
    "t_start": "2001-12-14T19:00:00Z",
    "t_end": "2001-12-14T20:00:00Z",
    "severity": "Critical",
    "final_verdict": "confirm",
    "llm_reason": "distance 4.9 sits at the 98th percentile of known rare events",
    "investigator_hypothesis": "isolated_deviation",
    "conformal_p": 0.000626,
    "routing_destination": "flight_director",
    "detector_version": "iforest-conformal-0.1.0",
}


def test_rendered_message_carries_the_operator_relevant_fields():
    body = json.dumps(slack.render(INCIDENT))
    for expected in (
        "Critical",
        "channel_20",
        "confirm",
        "isolated_deviation",
        "flight_director",
        "98th percentile of known rare events",
        "2001-12-14T19:00:00Z",
    ):
        assert expected in body, f"{expected} missing from the Slack message"


def test_conformal_p_is_rendered_readably_not_as_zero():
    """conformal_p spans orders of magnitude; plain float formatting prints
    0.000000 and destroys the only number that conveys how unusual the window is.
    """
    body = json.dumps(slack.render(dict(INCIDENT, conformal_p=6.26e-4)))
    assert "6.26e-04" in body
    assert "0.000000" not in body


def test_disputed_and_routing_anomaly_are_called_out_in_the_message():
    disputed = json.dumps(slack.render(dict(INCIDENT, final_verdict="disputed")))
    assert "disputed" in disputed.lower()
    # An operator must be able to tell WHY it is disputed. Under the old design
    # this meant "two models disagreed"; it now means the distance landed in the
    # band, and the message has to say the new thing.
    assert "band" in disputed.lower()

    flagged = json.dumps(slack.render(dict(INCIDENT, routing_anomaly=True)))
    assert "routing anomaly" in flagged.lower()


def test_a_model_disagreement_is_shown_as_audit_only():
    """The audit column reaches the operator, labelled so it cannot be mistaken
    for something that influenced the verdict.
    """
    body = json.dumps(slack.render(dict(INCIDENT, llm_verdict="reject"))).lower()
    assert "reject" in body
    assert "audit only" in body

    # Agreement is not worth a line: only disagreement is.
    agreeing = json.dumps(slack.render(dict(INCIDENT, llm_verdict="confirm"))).lower()
    assert "audit only" not in agreeing


def test_post_returns_delivered_on_2xx():
    notifier = _notifier_with()

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen", return_value=_Response()):
        assert notifier.post(INCIDENT) == slack.DELIVERED


def test_post_returns_failed_on_transport_error_without_raising():
    import urllib.error

    notifier = _notifier_with()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        assert notifier.post(INCIDENT) == slack.FAILED


def test_post_returns_failed_on_non_2xx():
    notifier = _notifier_with()

    class _Response:
        status = 500

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen", return_value=_Response()):
        assert notifier.post(INCIDENT) == slack.FAILED


def test_unconfigured_destination_is_reported_not_guessed():
    notifier = slack.SlackNotifier(project_id="p", client=object())
    notifier._resolved = True
    notifier._webhooks = {"flight_director": "https://hooks.example/a"}
    assert notifier.post(dict(INCIDENT, routing_destination="log")) == slack.SKIPPED


def test_every_destination_has_a_distinct_secret_id():
    """Three tiers, three secrets. A shared secret would send Critical pages to
    the log channel or vice versa.
    """
    assert set(slack.SECRET_IDS) == {d.value for d in RoutingDestination}
    assert len(set(slack.SECRET_IDS.values())) == 3


def test_webhook_url_never_appears_in_the_rendered_message():
    """The message is written to Firestore and to logs; a webhook in it would
    leak a bearer credential into both.
    """
    url = "https://hooks.slack.com/services/T000/B000/SECRETVALUE"
    notifier = _notifier_with(url)
    body = json.dumps(slack.render(INCIDENT))
    assert "SECRETVALUE" not in body
    assert "hooks.slack.com" not in body
    assert notifier  # notifier holds the url; the payload does not


@pytest.mark.parametrize(
    "status,expected",
    [
        (Verdict.CONFIRM, RoutingDestination.FLIGHT_DIRECTOR),
        (Verdict.REJECT, RoutingDestination.LOG),
        (Verdict.DISPUTED, RoutingDestination.LOG),
    ],
)
def test_verdict_to_destination_end_to_end(status, expected):
    incident, _ = run_graph(verifier_status=status.value)
    assert incident["routing_destination"] == expected.value


def test_incident_carries_the_window_it_is_about():
    """Regression: the Slack message rendered `Window ? -> ?` because
    IncidentDoc did not carry t_start/t_end. An alert naming a channel but not
    a time is not actionable.
    """
    incident, _ = run_graph(verifier_status="confirm")
    assert incident["t_start"] == DETECTION["t_start"]
    assert incident["t_end"] == DETECTION["t_end"]

    body = json.dumps(slack.render(incident))
    assert DETECTION["t_start"] in body
    assert "`?`" not in body
