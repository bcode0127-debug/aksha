"""Deliver an incident to the Slack channel its severity routes it to.

Webhook URLs live in Secret Manager and are read at cold start. They are never
logged, never echoed into a trace or an incident document, and never written to
a file in this repository: a Slack incoming webhook is a bearer credential, and
anyone holding one can post to that channel.

Delivery is best-effort by design (TRD section 9). A Slack outage must not lose
an incident that the graph has already reasoned about — the incident is recorded
either way and the attempt is reported as `delivered` or `failed`, so a silent
non-delivery is impossible to mistake for a successful one.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from aksha_agent.graph.schemas import RoutingDestination

logger = logging.getLogger(__name__)

PROJECT_ID_ENV = "GOOGLE_CLOUD_PROJECT"
POST_TIMEOUT_SECONDS = 10

# Secret Manager secret ids, one per destination tier (TRD section 7).
SECRET_IDS: dict[str, str] = {
    RoutingDestination.FLIGHT_DIRECTOR.value: "slack-flight-director",
    RoutingDestination.SUBSYSTEM_ENGINEER.value: "slack-subsystem",
    RoutingDestination.LOG.value: "slack-log",
}

SEVERITY_EMOJI = {"Critical": "🔴", "Caution": "🟠", "Advisory": "🔵"}

DELIVERED = "delivered"
FAILED = "failed"
SKIPPED = "not_configured"


def _fmt_p(value) -> str:
    """conformal_p spans several orders of magnitude; %f would print 0.000000."""
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return "n/a"


def render(incident: dict) -> dict:
    """Build the Slack payload. Readable at a glance — this is the demo shot.

    Kept deliberately plain: severity and verdict first, then the reasoning, then
    the numbers. Someone watching a recording should be able to tell what
    happened without pausing.
    """
    severity = str(incident.get("severity", "Advisory"))
    emoji = SEVERITY_EMOJI.get(severity, "⚪")
    destination = str(incident.get("routing_destination", "log"))
    verdict = str(incident.get("final_verdict") or "n/a")
    hypothesis = str(incident.get("investigator_hypothesis") or "n/a")
    reason = str(incident.get("llm_reason") or "").strip() or "_no reason recorded_"
    llm_verdict = incident.get("llm_verdict")

    headline = f"{emoji} {severity} — {incident.get('channel_id', 'unknown channel')}"
    flags = []
    if incident.get("routing_anomaly"):
        flags.append("⚠️ routing anomaly (unrecognised route, defaulted to log)")
    if verdict == "disputed":
        flags.append(
            "⚖️ disputed — the calibrated distance fell inside the uncertainty "
            "band; not escalated"
        )
    if llm_verdict and llm_verdict != verdict:
        # Audit only. The model's read did not affect the verdict or the route;
        # showing it lets an operator see when the two disagree.
        flags.append(f"🔍 model read this as `{llm_verdict}` (audit only, not applied)")

    lines = [
        f"*{headline}*",
        f"*Window* `{incident.get('t_start', '?')}` → `{incident.get('t_end', '?')}`",
        f"*Verdict* `{verdict}` (gate)  •  *Hypothesis* `{hypothesis}`",
        f"*conformal_p* `{_fmt_p(incident.get('conformal_p'))}` "
        f"(low = unlike nominal)  •  *routed to* `{destination}`",
        "",
        f"*Why:* {reason}",
    ]
    if flags:
        lines += ["", *flags]
    lines += [
        "",
        f"_incident_ `{incident.get('incident_id', '?')}`  •  "
        f"_detector_ `{incident.get('detector_version', '?')}`",
    ]
    body = "\n".join(lines)

    return {
        "text": f"{headline} — verdict {verdict} — routed to {destination}",
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": body}}],
    }


class SlackNotifier:
    """Posts incidents to the webhook for their routing destination.

    Secrets are resolved once and cached: a Secret Manager round trip per
    incident would put an avoidable dependency on the delivery path.
    """

    def __init__(self, project_id: str | None = None, client=None) -> None:
        self.project_id = project_id or os.environ.get(PROJECT_ID_ENV, "")
        self._client = client
        self._webhooks: dict[str, str] = {}
        self._resolved = False

    # --- secrets --------------------------------------------------------------

    def _secret_client(self):
        if self._client is None:
            from google.cloud import secretmanager

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def _resolve(self) -> None:
        """Read each webhook once. A missing secret disables that destination
        rather than failing the service: routing still records the outcome.
        """
        if self._resolved:
            return
        client = self._secret_client()
        for destination, secret_id in SECRET_IDS.items():
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": name})
                url = response.payload.data.decode("utf-8").strip()
                if url:
                    self._webhooks[destination] = url
            except Exception as exc:  # noqa: BLE001 - any failure disables the tier
                # The secret id is safe to log. The value never is.
                logger.warning(
                    "no webhook for %s (secret %s): %s",
                    destination,
                    secret_id,
                    type(exc).__name__,
                )
        self._resolved = True
        logger.info(
            "slack destinations configured: %s",
            sorted(self._webhooks) or "none",
        )

    @property
    def configured(self) -> list[str]:
        self._resolve()
        return sorted(self._webhooks)

    # --- delivery -------------------------------------------------------------

    def post(self, incident: dict) -> str:
        """Deliver one incident. Returns `delivered`, `failed` or `not_configured`.

        Never raises: the caller is a graph node whose job is to record the
        incident, and a Slack problem is not a reason to lose it.
        """
        self._resolve()
        destination = str(incident.get("routing_destination") or RoutingDestination.LOG.value)
        url = self._webhooks.get(destination)
        if not url:
            logger.warning("no webhook configured for destination %s", destination)
            return SKIPPED

        payload = json.dumps(render(incident)).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=POST_TIMEOUT_SECONDS) as response:
                if 200 <= response.status < 300:
                    logger.info(
                        "delivered incident %s to %s",
                        incident.get("incident_id"),
                        destination,
                    )
                    return DELIVERED
                logger.error(
                    "slack rejected incident %s for %s: HTTP %s",
                    incident.get("incident_id"),
                    destination,
                    response.status,
                )
                return FAILED
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Deliberately does not include the exception's full text: a URLError
            # raised by urllib can carry the request URL, i.e. the webhook.
            logger.error(
                "slack delivery failed for incident %s to %s: %s",
                incident.get("incident_id"),
                destination,
                type(exc).__name__,
            )
            return FAILED
