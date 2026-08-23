"""Pydantic models for every node boundary in the triage graph.

Each boundary is a schema, not a convention. ADK validates a node's input
against `input_schema` before the node runs — verified in the ADK spike
(ADR-013): a malformed payload fails at the receiving node's gate, before any
LLM call is made.

The load-bearing one is `ExplainerInput`. ADR-005 requires the explainer to be
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


class Verdict(str, Enum):
    """What the window IS: a fault operators should act on, or expected operation.

    This is the GATE's output. It is produced by comparing a calibrated distance
    against bounds loaded from the calibration artifact — no model is consulted.
    The explainer LLM emits the same enum as its own independent read, but that
    value is recorded for audit and never applied.

    Note especially what `disputed` means now. It is NOT "the model disagreed"
    and it is NOT "the investigator and the reviewer conflict". It means the
    calibrated distance landed inside the uncertainty band, so the number itself
    does not decide. That is a deterministic, reproducible statement about the
    evidence rather than a report on model behaviour.
    """

    CONFIRM = "confirm"      # a genuine fault: something is wrong with the spacecraft
    REJECT = "reject"        # unusual but expected operation, not a fault
    DISPUTED = "disputed"    # inside the calibrated band; the distance does not decide


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


# --- explain -------------------------------------------------------------------


class RecognitionEvidence(BaseModel):
    """Calibrated OUTLIERNESS: how far out on the tail this window sits.

    One-sided by construction — no anomaly-exemplar field — because that
    comparison measured near-chance (AUC 0.593 anomaly vs rare_event on 3,804
    vs 3,369 training windows) and dragged the explainer toward reject.

    Read as outlierness, NOT as resemblance. The calibration showed nominal
    windows sit closer to rare-event exemplars (median 1.05) than rare events
    sit to each other (median 4.37), so a small distance does not mean "this is
    a known rare event" — it means "this is unremarkable". What the distance
    tracks is distance from anything on record, which is what correlates with a
    fault. Describing it as recognition would describe something false.
    """

    distance_to_nearest_expected: float = Field(
        description="RMS standardised distance to the nearest known rare-event window."
    )
    percentile_among_rare_events: float = Field(
        ge=0.0,
        le=100.0,
        description=(
            "Where that distance falls in the distribution of known rare events' own "
            "nearest-neighbour distances. 50 means typical of a known expected pattern; "
            "95 means further out than 95% of them."
        ),
    )
    matched_exemplar: NeighbourWindow | None = Field(
        default=None, description="The nearest known expected window it was compared to."
    )
    reference_note: str = Field(
        default="",
        description="Calibration context, e.g. the median distance among known rare events.",
    )


class ExplainerInput(BaseModel):
    """ADR-005 enforcement, and the explainer's one-sided evidence.

    No `confidence` field exists, and `extra="forbid"` means one cannot be
    smuggled in: constructing this model with a confidence key raises
    `ValidationError`. The explainer therefore cannot anchor on how sure the
    investigator was, because that number never reaches it.

    Note what is ALSO absent: any anomaly exemplar. A "which is closer"
    comparison measured near-chance (AUC 0.593 anomaly vs rare_event) and
    dragged the model toward reject, so it is not supplied at all.
    """

    model_config = ConfigDict(extra="forbid")

    detection: DetectionSummary
    channel_history: ChannelHistorySummary
    recognition: RecognitionEvidence
    hypothesis: HypothesisKind
    implicated_channel: str
    evidence_refs: list[str] = Field(default_factory=list)


class ExplainerOutput(BaseModel):
    status: Verdict = Field(
        description=(
            "Your own independent read of what this window is. It is RECORDED "
            "FOR AUDIT and does not affect the outcome — a deterministic gate "
            "sets the verdict. Answer honestly rather than strategically. "
            "'confirm' = looks like a genuine fault. 'reject' = looks like "
            "routine or expected operation. 'disputed' = the evidence does not "
            "settle it."
        )
    )
    reason: str = Field(
        description=(
            "THE FIELD THAT MATTERS. One or two sentences an operator can act "
            "on, citing the distance, the percentile, and the feature z-scores "
            "that carry the story. This text is what appears in the alert."
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

    # Three fields, not one. The deterministic gate decides; the LLM's opinion
    # is recorded beside it rather than replacing it, so disagreement is visible
    # in the trace and on the dashboard instead of being silently resolved.
    # The gate decides. `final_verdict` and `gate_verdict` are therefore always
    # equal by construction; both are kept because the dashboard reads
    # `final_verdict` and dropping `gate_verdict` would make the audit column
    # below look like it had something to override.
    gate_verdict: Verdict | None = None
    final_verdict: Verdict | None = None
    gate_distance: float | None = None
    gate_threshold: float | None = None
    band_low: float | None = None
    band_high: float | None = None
    # AUDIT ONLY. The explainer's independent read and its explanation. The
    # verdict here never affects routing, severity or delivery — it is stored so
    # gate-vs-model disagreement stays measurable after the fact.
    llm_verdict: Verdict | None = None
    llm_reason: str | None = None

    severity: Severity
    routing_destination: RoutingDestination | None = None
    routing_outcome: str | None = None
    routing_timestamp_utc: str | None = None

    status: str = "open"
    routing_anomaly: bool = False
