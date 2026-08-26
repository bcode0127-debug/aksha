"""Read-only Streamlit telemetry-triage console over the pipeline's Firestore state.

Three panes, one screen, no scrolling to see the decision:
  1. Signal   -- the flagged channel's raw telemetry across the incident's
                 window, with the other lightweight channels as faint context.
  2. Decision -- the calibrated threshold-overlay bar (distance, band,
                 threshold, this incident's marker), the gate's verdict, and
                 the LLM's read as a muted audit line.
  3. Incident strip -- recent incidents, click to load into panes 1 and 2.

Read-only by construction: every Firestore call is `.stream()`/`.get()`.
Nothing here writes, deletes, or authenticates.

    streamlit run aksha_agent/dashboard/app.py

Same Firestore client construction as the two Cloud Run services
(aksha_agent/infra/*/*.py): `firestore.Client(project=...)`, ADC locally.

Pane 1 reads from data/processed/dashboard_signal_cache.parquet, built by
scripts/precompute_signal_cache.py -- see that script's docstring for why
(loading Mission2 raw channels is a full-series read, not windowed, and too
slow to do per incident per interaction). Run it once after new incidents
land in Firestore; incidents not yet in the cache show a clearly labelled
"not cached" state in Pane 1 rather than a slow live load.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from google.cloud import firestore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
INCIDENT_LIMIT = 50
SIGNAL_CACHE_PATH = REPO_ROOT / "data" / "processed" / "dashboard_signal_cache.parquet"
FINAL_REPORT_PATH = REPO_ROOT / "eval" / "outputs" / "final_report.json"

BG = "#0E1117"
FG = "#E8E8E8"
MUTED = "#8A8A8A"
GRID = "#2A2E37"

# ISA-18.2-inspired: every state carries color + text label + a distinct
# shape/symbol, never color alone. Desaturated -- this is a console, not a
# warning light.
SEVERITY_STYLE = {
    "Critical": {"color": "#C75450", "symbol": "■", "marker": "s"},   # filled square
    "Caution":  {"color": "#C9A227", "symbol": "▲", "marker": "^"},   # triangle
    "Advisory": {"color": "#5FA88A", "symbol": "●", "marker": "o"},   # circle
}
VERDICT_STYLE = {
    "confirm":  {"color": "#C75450", "symbol": "■", "marker": "s"},
    "disputed": {"color": "#C9A227", "symbol": "▲", "marker": "^"},
    "reject":   {"color": "#5FA88A", "symbol": "●", "marker": "o"},
}
NO_DATA_STYLE = {"color": "#8A8A8A", "symbol": "○", "marker": "o"}  # open circle, gray

EVALUATION_MD_FALLBACK = {
    "source": "docs/EVALUATION.md (static fallback: eval/outputs/final_report.json not found locally)",
    "gate_confirm_rate": 0.885, "gate_confirm_count": 177,
    "llm_confirm_rate": 0.515, "llm_confirm_count": 103, "n_holdout": 200,
    "golden_set_agreement": "18/18",
}


@st.cache_resource
def get_client() -> firestore.Client:
    return firestore.Client(project=PROJECT_ID)


@st.cache_data(ttl=30)
def load_incidents(limit: int) -> list[dict]:
    """Recent incidents, newest first. Docs missing timestamp_utc (a handful
    of pre-pipeline test fragments) are excluded by Firestore's order_by
    semantics, not filtered here.
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


@st.cache_data
def load_signal_cache() -> pd.DataFrame | None:
    if not SIGNAL_CACHE_PATH.exists():
        return None
    return pd.read_parquet(SIGNAL_CACHE_PATH)


def load_evidence() -> dict:
    if FINAL_REPORT_PATH.exists():
        report = json.loads(FINAL_REPORT_PATH.read_text())
        gate = report["gate_vs_llm"]
        golden = report["golden_set"]
        return {
            "source": f"eval/outputs/final_report.json (commit {report.get('commit', '?')[:7]})",
            "gate_confirm_rate": gate["gate_confirm_rate"], "gate_confirm_count": gate["gate_confirm_count"],
            "llm_confirm_rate": gate["llm_confirm_rate"], "llm_confirm_count": gate["llm_confirm_count"],
            "n_holdout": gate["n_windows"], "golden_set_agreement": golden["scored_agreement"],
        }
    return EVALUATION_MD_FALLBACK


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        html, body, [class*="css"] { font-size: 17px; }
        h1 { font-size: 2.1rem !important; }
        h2, .aksha-pane-title { font-size: 1.3rem !important; }
        .aksha-badge {
            display: inline-block; padding: 0.1em 0.6em; border-radius: 999px;
            font-weight: 700; font-size: 0.9em; color: #0B0B0B; white-space: nowrap;
        }
        .aksha-strip-row { border-bottom: 1px solid #2A2E37; padding: 0.25em 0; font-size: 0.95em; }
        .aksha-disagree {
            background-color: #3A1414; border: 2px solid #C75450; border-radius: 8px;
            padding: 0.6em 0.9em; font-size: 1.05em; font-weight: 700; color: #F0A8A5; margin-top: 0.5em;
        }
        .aksha-agree {
            background-color: #14251A; border: 2px solid #5FA88A; border-radius: 8px;
            padding: 0.5em 0.9em; font-size: 0.95em; font-weight: 600; color: #A6D9C2; margin-top: 0.5em;
        }
        .aksha-audit {
            color: #8A8A8A; font-size: 0.9em; font-style: italic; margin-top: 0.4em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(label: str | None, style_map: dict) -> str:
    style = style_map.get(label or "", NO_DATA_STYLE)
    text = label or "no data"
    return f'<span class="aksha-badge" style="background-color:{style["color"]}">{style["symbol"]} {text}</span>'


def _dark_axes(ax) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)


# --- Pane 1: Signal ---------------------------------------------------------

def render_signal_pane(incident: dict) -> None:
    channel = incident.get("channel_id", "?")
    t_start, t_end = incident.get("t_start", "?"), incident.get("t_end", "?")
    st.markdown(f'<div class="aksha-pane-title">Signal: {channel}</div>', unsafe_allow_html=True)
    st.caption(f"{t_start} → {t_end} UTC")

    cache = load_signal_cache()
    incident_id = incident["_id"]
    sub = cache[cache["incident_id"] == incident_id] if cache is not None else pd.DataFrame()

    if sub.empty:
        st.markdown(
            '<div style="background:#1A1D23; border:1px dashed #444; border-radius:8px; '
            'padding:2.5em; text-align:center; color:#8A8A8A;">'
            "No cached signal data for this incident.<br>"
            "<code>python3 scripts/precompute_signal_cache.py</code> to include it."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    flagged = sub[sub["is_flagged"]]
    others = sub[~sub["is_flagged"]]

    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=110)
    fig.patch.set_facecolor(BG)
    _dark_axes(ax)

    for ch_name, grp in others.groupby("channel_id"):
        ax.plot(grp["timestamp"], grp["value"], color="#555555", linewidth=0.7, alpha=0.55, zorder=1)

    if not flagged.empty:
        window_start = flagged["window_start"].iloc[0]
        window_end = flagged["window_end"].iloc[0]
        ax.axvspan(window_start, window_end, color="#4FA3FF", alpha=0.12, zorder=0)
        ax.plot(flagged["timestamp"], flagged["value"], color="#4FA3FF", linewidth=2.0, zorder=3, label=channel)
        ax.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)

    ax.set_ylabel("value")
    fig.autofmt_xdate(rotation=20)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(
        f"Bold blue: {channel} (flagged). Faint gray: the other "
        f"{others['channel_id'].nunique() if not others.empty else 0} lightweight channels, for context. "
        "Shaded band: the incident window."
    )


# --- Pane 2: Decision --------------------------------------------------------

def render_threshold_bar(distance, threshold, band_low, band_high, gate_verdict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 1.3), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xmax = max(v for v in (distance, threshold, band_high) if v is not None) * 1.25
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    if band_low is not None and band_high is not None:
        ax.axvspan(band_low, band_high, color="#C9A227", alpha=0.22, zorder=0, label="ambiguous band")
    if threshold is not None:
        ax.axvline(threshold, color="#EEEEEE", linestyle="--", linewidth=1.3, zorder=2)
        ax.text(threshold, 1.08, f"threshold {threshold:.3f}", color=MUTED, fontsize=7.5,
                ha="center", transform=ax.get_xaxis_transform())

    style = VERDICT_STYLE.get(gate_verdict, NO_DATA_STYLE)
    if distance is not None:
        ax.scatter([distance], [0.5], s=220, marker=style["marker"], color=style["color"],
                   edgecolor="white", linewidth=1.3, zorder=5)
        ax.text(distance, -0.35, f"{distance:.3f}", color=FG, fontsize=8, ha="center",
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("calibrated distance", fontsize=8, color=MUTED)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_decision_pane(incident: dict) -> None:
    st.markdown('<div class="aksha-pane-title">Decision</div>', unsafe_allow_html=True)

    gate_verdict = incident.get("gate_verdict")
    llm_verdict = incident.get("llm_verdict")
    distance = incident.get("gate_distance")
    threshold = incident.get("gate_threshold")
    band_low, band_high = incident.get("band_low"), incident.get("band_high")

    if None not in (distance, threshold, band_low, band_high):
        render_threshold_bar(distance, threshold, band_low, band_high, gate_verdict)
    else:
        st.caption("No calibration data recorded for this incident.")

    severity = incident.get("severity")
    st.markdown(
        f"**Gate verdict:** {badge_html(gate_verdict, VERDICT_STYLE)}&nbsp;&nbsp;"
        f"**Severity:** {badge_html(severity, SEVERITY_STYLE)}",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Routed to:** `{incident.get('routing_destination', '-')}`  "
                f"&middot; **Delivery:** `{incident.get('routing_outcome', '--')}`")

    reason = incident.get("llm_reason") or "(no reason recorded)"
    st.markdown(
        f'<div class="aksha-audit">model read (audit only, not applied): '
        f"<b>{llm_verdict or 'n/a'}</b>: {reason}</div>",
        unsafe_allow_html=True,
    )

    disagree = gate_verdict is not None and llm_verdict is not None and gate_verdict != llm_verdict
    if disagree:
        st.markdown(
            f'<div class="aksha-disagree">⚠ DISAGREEMENT: gate: '
            f'<b>{VERDICT_STYLE.get(gate_verdict, NO_DATA_STYLE)["symbol"]} {gate_verdict}</b>, '
            f'model: <b>{llm_verdict}</b>. Gate\'s verdict routed; model\'s read is audit only.</div>',
            unsafe_allow_html=True,
        )
    elif gate_verdict is not None and llm_verdict is not None:
        st.markdown(
            f'<div class="aksha-agree">✓ Gate and model agree: {gate_verdict}</div>',
            unsafe_allow_html=True,
        )


# --- Pane 3: Incident strip --------------------------------------------------

def render_incident_strip(incidents: list[dict], selected_id: str | None) -> None:
    st.markdown('<div class="aksha-pane-title">Recent incidents</div>', unsafe_allow_html=True)
    if not incidents:
        st.info("No incidents in Firestore yet. Publish one via scripts/publish_stub.py.")
        return

    header = st.columns([1.8, 1.0, 1.0, 1.0, 1.3, 0.7])
    for col, text in zip(header, ["Time (UTC)", "Channel", "Severity", "Verdict", "Destination", ""]):
        col.markdown(f"**{text}**")

    for incident in incidents:
        cols = st.columns([1.8, 1.0, 1.0, 1.0, 1.3, 0.7])
        is_selected = incident["_id"] == selected_id
        # Pre-gate incidents (no gate_verdict at all -- test fragments from
        # before PR #14's deterministic gate) carry no decision to show, so
        # they're dimmed rather than left visually identical to a real one.
        has_gate_data = incident.get("gate_verdict") is not None
        row_style = "" if has_gate_data else "opacity:0.45;"
        prefix = "▶ " if is_selected else ""
        cols[0].markdown(f'<span style="{row_style}">{prefix}{incident.get("timestamp_utc", "-")}</span>', unsafe_allow_html=True)
        cols[1].markdown(f'<span style="{row_style}">{incident.get("channel_id", "-")}</span>', unsafe_allow_html=True)
        cols[2].markdown(badge_html(incident.get("severity"), SEVERITY_STYLE), unsafe_allow_html=True)
        cols[3].markdown(badge_html(incident.get("final_verdict"), VERDICT_STYLE), unsafe_allow_html=True)
        cols[4].markdown(f'<span style="{row_style}">{incident.get("routing_destination", "-")}</span>', unsafe_allow_html=True)
        if cols[5].button("Load", key=f"select_{incident['_id']}", disabled=is_selected):
            st.session_state["selected_incident"] = incident["_id"]
            st.rerun()


# --- Evidence (kept, compact, below the fold) --------------------------------

def render_evidence_strip() -> None:
    ev = load_evidence()
    st.caption(
        f"Gate {ev['gate_confirm_rate']:.1%} ({ev['gate_confirm_count']}/{ev['n_holdout']}) vs. "
        f"LLM alone {ev['llm_confirm_rate']:.1%} ({ev['llm_confirm_count']}/{ev['n_holdout']}) on held-out faults "
        f"· golden set {ev['golden_set_agreement']} · source: {ev['source']}"
    )


@st.fragment(run_every=30)
def render_body() -> None:
    """Everything that needs to see new Firestore writes without a manual
    reload. Wrapped in a fragment with run_every=30 so the 30s refresh in the
    caption is an actual mechanism, not just a claim: Streamlit reruns this
    function on that timer on its own, independent of user interaction --
    @st.cache_data(ttl=30) above only expires a cached value, it does not by
    itself cause anything to re-run and re-fetch it.
    """
    incidents = load_incidents(INCIDENT_LIMIT)
    selected_id = st.session_state.get("selected_incident")
    selected = next((i for i in incidents if i["_id"] == selected_id), None)
    if selected is None and incidents:
        # Default to the newest incident that actually has gate data, not
        # simply the newest overall -- incidents/ carries a long tail of
        # pre-gate test fragments (predating PR #14's deterministic gate)
        # with no gate_verdict at all, and opening the Decision pane on one
        # of those means it opens empty every time.
        selected = next((i for i in incidents if i.get("gate_verdict") is not None), incidents[0])
        st.session_state["selected_incident"] = selected["_id"]

    top_left, top_right = st.columns([65, 35])
    with top_left:
        if selected:
            render_signal_pane(selected)
        else:
            st.info("No incidents to show.")
    with top_right:
        if selected:
            render_decision_pane(selected)

    st.divider()
    render_incident_strip(incidents, selected["_id"] if selected else None)
    render_evidence_strip()


def main() -> None:
    st.set_page_config(page_title="AKSHA telemetry triage console", layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    plt.rcParams.update({"text.color": FG, "axes.edgecolor": GRID, "axes.labelcolor": MUTED})

    st.title("AKSHA: telemetry triage console")
    st.caption(f"Firestore project: `{PROJECT_ID}` · read-only · refreshes every 30s")

    render_body()


if __name__ == "__main__":
    main()
