"""Pydantic models for every node boundary in the triage graph.

Each boundary is a schema, not a convention. ADK validates a node's input
against `input_schema` before the node runs — verified in the ADK spike
(ADR-013): a malformed payload fails at the receiving node's gate, before any
LLM call is made.

The load-bearing one is `VerifierInput`. ADR-005 requires the verifier to be
structurally unable to see the investigator's confidence, and this is where
"structurally" is cashed out: the model has no `confidence` field and forbids
extras, so passing one raises `ValidationError` rather than being quietly
ignored. That is enforcement, not documentation.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class HypothesisKind(str, Enum):
    """The investigator's verdict, bounded to a closed set.

    Free-text hypotheses cannot be routed on, counted, or compared across
    incidents, and they let the model invent a category rather than commit to
    one. A closed enum forces a decision the rest of the graph can act on.
    """

    COMMAND_INDUCED = "command_induced"
    CORRELATED_CHANNEL = "correlated_channel"
    ISOLATED_DEVIATION = "isolated_deviation"
    GAP_ARTIFACT = "gap_artifact"
    SENSOR_NOISE = "sensor_noise"


class VerifierStatus(str, Enum):
    """The verifier's adjudication: genuine fault, or expected operation?

    These do NOT mean "the hypothesis is supported / contradicted". Asking
    whether the evidence supports the hypothesis makes the verifier unable to
    discriminate: a rare-but-expected event IS statistically deviant, so
    "supported" is the correct answer for it too, and everything gets confirmed.

    The verifier instead decides what the window IS — a fault that operators
    should act on, or unusual-but-expected behaviour that they should not be
    woken for.
    """

    CONFIRM = "confirm"      # a genuine fault: something is wrong with the spacecraft
    REJECT = "reject"        # unusual but expected operation, not a fault
    DISPUTED = "disputed"    # the evidence does not settle it either way


class Severity(str, Enum):
    CRITICAL = "Critical"
    CAUTION = "Caution"
    ADVISORY = "Advisory"


class RoutingDestination(str, Enum):
    FLIGHT_DIRECTOR = "flight_director"
    SUBSYSTEM_ENGINEER = "subsystem_engineer"
    LOG = "log"


# --- what crosses the core/agent boundary ------------------------------------


class DetectionSummary(BaseModel):
    """The DetectionResult as it reaches the graph.

    `conformal_p` is a conformal p-value: LOW means anomalous. `threshold` is a
    RAW SCORE threshold and is not comparable to `conformal_p` — see
    `aksha_agent.graph.workflow.compute_severity`.

    `features` is a mapping of named scalars. No raw telemetry arrays cross
    this boundary (ADR-003).
    """

    fragment_id: str
    channel_id: str
    t_start: str
    t_end: str
    score: float
    threshold: float
    conformal_p: float = Field(ge=0.0, le=1.0)
    detector_version: str
    features: dict[str, float] = Field(default_factory=dict)


class ChannelStat(BaseModel):
    """One feature's nominal envelope on a channel, from the training split."""

    feature: str
    nominal_mean: float
    nominal_std: float
    window_value: float
    z_score: float


class ChannelHistorySummary(BaseModel):
    """How this window sits against the channel's own nominal history.

    A summary of scalars, never a series (ADR-003).
    """

    channel_id: str
    reference_windows: int
    most_deviant: list[ChannelStat] = Field(default_factory=list)


class NeighbourWindow(BaseModel):
    """A nearby labelled window, as a feature summary rather than a series."""

    channel_id: str
    window_start: str
    label: str
    distance: float
    features: dict[str, float] = Field(default_factory=dict)


# --- investigate --------------------------------------------------------------


class InvestigateInput(BaseModel):
    detection: DetectionSummary
    channel_history: ChannelHistorySummary
    nearest_labeled: list[NeighbourWindow] = Field(default_factory=list)


class InvestigateOutput(BaseModel):
    hypothesis: HypothesisKind = Field(
        description=(
            "Pick the explanation the evidence actually supports. "
            "'command_induced': telecommand activity coincides with the deviation "
            "(check tc_count_in_window and seconds_since_last_tc against nominal). "
            "'correlated_channel': the window moves with related channels (check "
            "the mahalanobis z-score, which is the cross-channel signal). "
            "'gap_artifact': missing data explains the shape (check gap_fraction "
            "and max_gap_seconds). "
            "'isolated_deviation': this channel alone departs from its history, "
            "with no command or cross-channel or gap explanation. "
            "'sensor_noise': within plausible noise for this channel."
        )
    )
    implicated_channel: str
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Feature names with their z-scores or values that justify the choice.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


# --- verify -------------------------------------------------------------------


class VerifierInput(BaseModel):
    """ADR-005 enforcement.

    No `confidence` field exists, and `extra="forbid"` means one cannot be
    smuggled in: constructing this model with a confidence key raises
    `ValidationError`. The verifier therefore cannot anchor on how sure the
    investigator was, because that number never reaches it.
    """

    model_config = ConfigDict(extra="forbid")

    detection: DetectionSummary
    channel_history: ChannelHistorySummary
    nearest_labeled: list[NeighbourWindow] = Field(default_factory=list)
    hypothesis: HypothesisKind
    implicated_channel: str
    evidence_refs: list[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    status: VerifierStatus = Field(
        description=(
            "Adjudicate what this window IS, not whether the hypothesis sounds "
            "plausible. 'confirm' = a genuine fault requiring operator attention. "
            "'reject' = unusual but expected operation (a manoeuvre, a commanded "
            "change, a known rare mode) — deviant from nominal, but not a fault. "
            "'disputed' = the evidence genuinely does not settle it."
        )
    )
    reason: str = Field(
        description=(
            "One or two sentences citing the specific exemplar distances and "
            "feature values that decided it. State which labelled exemplar the "
            "window most resembles and why."
        )
    )


# --- filing -------------------------------------------------------------------


class IncidentDoc(BaseModel):
    """The incident as written to Firestore.

    No ground-truth label appears here: labels exist only in the eval harness
    (TRD section 6).
    """

    incident_id: str
    timestamp_utc: str

    channel_id: str
    fragment_id: str
    detector_version: str

    # The window this incident is about. Without these an alert says a channel
    # is faulty but not when, which is not actionable — and the Slack message
    # rendered them as "?" until they were carried here.
    t_start: str | None = None
    t_end: str | None = None

    anomaly_score: float
    threshold: float
    conformal_p: float

    features_summary: dict[str, float] = Field(default_factory=dict)

    investigator_hypothesis: HypothesisKind | None = None
    investigator_confidence: float | None = None
    implicated_channel: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)

    verifier_status: VerifierStatus | None = None
    verifier_reason: str | None = None

    severity: Severity
    routing_destination: RoutingDestination | None = None
    routing_outcome: str | None = None
    routing_timestamp_utc: str | None = None

    status: str = "open"
    routing_anomaly: bool = False
