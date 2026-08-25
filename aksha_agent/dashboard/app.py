"""Read-only Streamlit dashboard over the triage pipeline's Firestore state.

Three sections: recent incidents (incidents/), the full per-node trace and
gate-vs-LLM audit block for whichever one is selected (traces/{id}/steps/{n}),
and a static evidence panel read from the eval harness.

Read-only by construction: every Firestore call below is `.stream()` or
`.get()`. Nothing here writes, deletes, or authenticates -- it's a local
demo view, not a service.

    streamlit run aksha_agent/dashboard/app.py

Same Firestore client construction as the two Cloud Run services
(aksha_agent/infra/*/*.py): `firestore.Client(project=...)`, ADC locally via
the runtime service account (`gcloud auth application-default login`, or
`GOOGLE_APPLICATION_CREDENTIALS` for the aksha-runtime key) -- nothing
dashboard-specific to configure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from google.cloud import firestore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
INCIDENT_LIMIT = 50

SEVERITY_COLOR = {
    "Critical": "#FF5C5C",
    "Caution": "#FFB020",
    "Advisory": "#4FA3FF",
}
SEVERITY_ORDER = {"Critical": 0, "Caution": 1, "Advisory": 2}

FINAL_REPORT_PATH = REPO_ROOT / "eval" / "outputs" / "final_report.json"

# Static fallback, sourced from docs/EVALUATION.md, for when the gitignored
# eval/outputs/final_report.json hasn't been regenerated locally. Numbers
# must match that doc; if they ever diverge, docs/EVALUATION.md is the one
# that's actually re-verified against a live run (tests/test_evaluation_report.py),
# so trust that over this fallback.
EVALUATION_MD_FALLBACK = {
    "source": "docs/EVALUATION.md (static fallback -- eval/outputs/final_report.json not found locally)",
    "gate_confirm_rate": 0.885,
    "gate_confirm_count": 177,
    "llm_confirm_rate": 0.515,
    "llm_confirm_count": 103,
    "n_holdout": 200,
    "golden_set_agreement": "18/18",
    "detector_metrics": {
        "accuracy": 0.979, "precision": 0.704, "recall": 0.650,
        "f1": 0.676, "mcc": 0.666, "auc_roc": 0.853, "auc_pr": 0.653,
    },
    "auc_fault_vs_expected": 0.795,
    "auc_anomaly_vs_rare_event": 0.610,
}


@st.cache_resource
def get_client() -> firestore.Client:
    return firestore.Client(project=PROJECT_ID)


@st.cache_data(ttl=30)
def load_incidents(limit: int) -> list[dict]:
    """Recent incidents, newest first.

    Ordering by timestamp_utc means any incident doc missing that field
    (a handful of pre-this-pipeline test fragments) is excluded by
    Firestore's own order_by semantics, not filtered here -- an incomplete
    record shouldn't outrank a real one at the top of a demo list.
    """
    db = get_client()
    query = (
        db.collection("incidents")
        .order_by("timestamp_utc", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    return [d.to_dict() | {"_id": d.id} for d in query.stream()]


@st.cache_data(ttl=30)
def load_trace(incident_id: str) -> list[dict]:
    db = get_client()
    query = db.collection("traces").document(incident_id).collection("steps").order_by("step")
    return [d.to_dict() for d in query.stream()]


def load_evidence() -> dict:
    if FINAL_REPORT_PATH.exists():
        report = json.loads(FINAL_REPORT_PATH.read_text())
        gate = report["gate_vs_llm"]
        detector = report["detector"]["metrics_rare_event_union_anomaly"]
        golden = report["golden_set"]
        auc = report["separability_auc"]
        return {
            "source": f"eval/outputs/final_report.json (commit {report.get('commit', '?')[:7]})",
            "gate_confirm_rate": gate["gate_confirm_rate"],
            "gate_confirm_count": gate["gate_confirm_count"],
            "llm_confirm_rate": gate["llm_confirm_rate"],
            "llm_confirm_count": gate["llm_confirm_count"],
            "n_holdout": gate["n_windows"],
            "golden_set_agreement": golden["scored_agreement"],
            "detector_metrics": {
                "accuracy": detector["accuracy"], "precision": detector["precision"],
                "recall": detector["recall"], "f1": detector["f1"], "mcc": detector["mcc"],
                "auc_roc": detector["auc_roc"], "auc_pr": detector["auc_pr"],
            },
            "auc_fault_vs_expected": auc["auc_anomaly_vs_nominal_and_rare_event"],
            "auc_anomaly_vs_rare_event": auc["auc_anomaly_vs_rare_event_only"],
        }
    return EVALUATION_MD_FALLBACK


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        html, body, [class*="css"] { font-size: 18px; }
        h1 { font-size: 2.4rem !important; }
        h2 { font-size: 1.8rem !important; }
        h3 { font-size: 1.4rem !important; }
        .aksha-badge {
            display: inline-block; padding: 0.15em 0.7em; border-radius: 999px;
            font-weight: 700; font-size: 0.95em; color: #0B0B0B;
        }
        .aksha-row {
            border-bottom: 1px solid #333; padding: 0.5em 0; font-size: 1.05em;
        }
        .aksha-disagree {
            background-color: #3A1414; border: 2px solid #FF5C5C; border-radius: 8px;
            padding: 0.8em 1em; font-size: 1.2em; font-weight: 700; color: #FF9C9C;
        }
        .aksha-agree {
            background-color: #14251A; border: 2px solid #3DDC84; border-radius: 8px;
            padding: 0.8em 1em; font-size: 1.1em; font-weight: 700; color: #8CF0B4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(severity: str | None) -> str:
    color = SEVERITY_COLOR.get(severity or "", "#888888")
    label = severity or "unknown"
    return f'<span class="aksha-badge" style="background-color:{color}">{label}</span>'


def render_incident_list(incidents: list[dict]) -> None:
    st.header("Incidents")
    if not incidents:
        st.info("No incidents in Firestore yet -- publish one via scripts/publish_stub.py.")
        return

    header = st.columns([2, 1.3, 1.1, 1.1, 1.4, 1.2, 0.8])
    for col, text in zip(header, ["Timestamp (UTC)", "Channel", "Severity", "Verdict", "Routed to", "Delivery", ""]):
        col.markdown(f"**{text}**")

    for incident in incidents:
        cols = st.columns([2, 1.3, 1.1, 1.1, 1.4, 1.2, 0.8])
        cols[0].markdown(str(incident.get("timestamp_utc", "—")))
        cols[1].markdown(str(incident.get("channel_id", "—")))
        cols[2].markdown(severity_badge(incident.get("severity")), unsafe_allow_html=True)
        cols[3].markdown(str(incident.get("final_verdict", "—")))
        cols[4].markdown(str(incident.get("routing_destination", "—")))
        cols[5].markdown(str(incident.get("routing_outcome", "—")))
        if cols[6].button("View", key=f"select_{incident['_id']}"):
            st.session_state["selected_incident"] = incident["_id"]


def render_incident_detail(incident: dict) -> None:
    st.header(f"Incident: {incident['_id']}")

    top = st.columns(4)
    top[0].metric("Channel", incident.get("channel_id", "—"))
    top[1].markdown(f"**Severity**<br>{severity_badge(incident.get('severity'))}", unsafe_allow_html=True)
    top[2].metric("Final verdict", incident.get("final_verdict", "—"))
    top[3].metric("Delivery", incident.get("routing_outcome", "—"))

    st.subheader("Gate vs. LLM -- the audit column")
    gate_verdict = incident.get("gate_verdict")
    llm_verdict = incident.get("llm_verdict")
    disagree = gate_verdict is not None and llm_verdict is not None and gate_verdict != llm_verdict

    gcols = st.columns([1, 1, 1.4, 1, 1])
    gcols[0].metric("Gate distance", incident.get("gate_distance", "—"))
    gcols[1].metric("Threshold", incident.get("gate_threshold", "—"))
    band_low, band_high = incident.get("band_low"), incident.get("band_high")
    gcols[2].markdown(
        f"**Band**<br><span style='font-size:1.6em'>[{band_low}, {band_high}]</span>"
        if band_low is not None else "**Band**<br>—",
        unsafe_allow_html=True,
    )
    gcols[3].metric("Gate verdict", gate_verdict or "—")
    gcols[4].metric("LLM verdict", llm_verdict or "—")

    if disagree:
        st.markdown(
            '<div class="aksha-disagree">&#9888; DISAGREEMENT -- the gate decided '
            f'<b>{gate_verdict}</b>, the model read it as <b>{llm_verdict}</b>. '
            "The gate's verdict is what routed; the model's read is audit only.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="aksha-agree">&#10003; Gate and LLM agree: {gate_verdict or "n/a"}</div>',
            unsafe_allow_html=True,
        )

    if incident.get("llm_reason"):
        st.markdown(f"**Explainer reasoning:** {incident['llm_reason']}")

    st.subheader("Full trace")
    steps = load_trace(incident["_id"])
    if not steps:
        st.warning("No trace steps found for this incident.")
    for step in steps:
        node = step.get("node", "?")
        with st.expander(f"step {step.get('step')} -- {node}", expanded=False):
            shown = {k: v for k, v in step.items() if k not in ("step", "node")}
            st.json(shown)


def render_evidence_panel() -> None:
    st.header("Evidence panel")
    ev = load_evidence()
    st.caption(f"Source: {ev['source']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Gate confirm rate (200 held-out faults)",
              f"{ev['gate_confirm_rate']:.1%}", f"{ev['gate_confirm_count']}/{ev['n_holdout']}")
    c2.metric("LLM alone confirm rate",
              f"{ev['llm_confirm_rate']:.1%}", f"{ev['llm_confirm_count']}/{ev['n_holdout']}")
    c3.metric("Golden set (22 windows)", ev["golden_set_agreement"], "scored agreement")

    d = ev["detector_metrics"]
    st.markdown(
        f"**Detector** (rare_event &cup; anomaly positive class): "
        f"accuracy {d['accuracy']:.3f} · precision {d['precision']:.3f} · recall {d['recall']:.3f} · "
        f"F1 {d['f1']:.3f} · MCC {d['mcc']:.3f} · AUC-ROC {d['auc_roc']:.3f} · AUC-PR {d['auc_pr']:.3f}"
    )
    st.markdown(
        f"**Separability AUC:** fault-vs-expected {ev['auc_fault_vs_expected']:.3f} · "
        f"anomaly-vs-rare_event {ev['auc_anomaly_vs_rare_event']:.3f}"
    )


def main() -> None:
    st.set_page_config(page_title="AKSHA triage dashboard", layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    st.title("AKSHA -- triage dashboard")
    st.caption(f"Firestore project: `{PROJECT_ID}` · read-only · refreshes every 30s")

    incidents = load_incidents(INCIDENT_LIMIT)
    render_incident_list(incidents)

    st.divider()
    selected_id = st.session_state.get("selected_incident")
    selected = next((i for i in incidents if i["_id"] == selected_id), None) if selected_id else None
    if selected:
        render_incident_detail(selected)
    else:
        st.caption("Select an incident above (\"View\") to see its trace and audit block.")

    st.divider()
    render_evidence_panel()


if __name__ == "__main__":
    main()
