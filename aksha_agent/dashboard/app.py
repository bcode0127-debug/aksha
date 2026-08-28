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
FUNNEL_STATS_PATH = REPO_ROOT / "aksha_core" / "artifacts" / "mission2_funnel_stats.json"

# Human-readable labels for the real graph node names that appear in
# traces/{id}/steps/{n} -- see aksha_agent/graph/workflow.py. Two entries are
# deliberately dual-labelled: investigate and explain are LlmAgent nodes with
# no trace step of their own (ADK has no node-level callback for agent nodes,
# ADR-013 spike; aksha_agent/graph/workflow.py's trace() calls are only in
# function nodes), so their real output is stored in the FOLLOWING function
# node's trace payload. The label says so rather than pretending there is a
# separate stored step.
NODE_LABELS: dict[str, dict] = {
    "prepare_context": {
        "title": "1. Prepare context", "model": False,
        "what": "Loaded the channel's reference history and nearest labeled neighbors.",
    },
    "assemble_explainer_input": {
        "title": "2. Investigate -> 3. Assemble explainer input", "model": True,
        "what": "Gemini investigated (hypothesis, implicated channel, evidence), then the "
                "result was packaged for the explainer. investigate has no trace step of its "
                "own (ADR-013): its output is stored in this step's payload.",
    },
    "verification_gate": {
        "title": "4. Explain -> 5. Verification gate", "model": True,
        "what": "Gemini explained the window in operator language (audit only), then the "
                "deterministic gate compared the calibrated distance against the threshold "
                "and band. explain has no trace step of its own, same reason as investigate.",
    },
    "route_by_status": {
        "title": "6. Route: by gate verdict", "model": False,
        "what": "Routed on the gate's verdict alone -- confirm, disputed, or reject.",
    },
    "file_report": {"title": "6. Route: file report", "model": False, "what": "Filed the incident, computed severity from conformal_p."},
    "file_report_rejected": {"title": "6. Route: file report (rejected)", "model": False, "what": "Filed the incident as expected operation -- logged, not escalated."},
    "file_report_disputed": {"title": "6. Route: file report (disputed)", "model": False, "what": "Filed the incident as disputed -- logged, not escalated."},
    "file_report_unroutable": {"title": "6. Route: file report (unroutable)", "model": False, "what": "Gate emitted a verdict no branch claimed (DEFAULT_ROUTE) -- filed and flagged anyway."},
    "notify_flight_director": {"title": "6. Route: deliver", "model": False, "what": "Delivered to the flight director via the slack-flight-director webhook."},
    "notify_subsystem_engineer": {"title": "6. Route: deliver", "model": False, "what": "Delivered to the subsystem engineer via the slack-subsystem webhook."},
    "record_to_log": {"title": "6. Route: deliver", "model": False, "what": "Recorded to the log channel via the slack-log webhook."},
    "record_unroutable": {"title": "6. Route: deliver", "model": False, "what": "Severity router's DEFAULT_ROUTE -- recorded to the log channel anyway."},
}

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


@st.cache_data(ttl=30)
def load_signal_cache() -> pd.DataFrame | None:
    if not SIGNAL_CACHE_PATH.exists():
        return None
    return pd.read_parquet(SIGNAL_CACHE_PATH)


@st.cache_data
def load_funnel_stats() -> dict | None:
    """The funnel strip's three static numbers, precomputed by
    scripts/build_funnel_stats.py from the real Mission2 adapter output and
    the committed detector -- not read fresh on every page load (that would
    mean shipping the 27 MB feature parquet into the deploy image and
    rescoring 168k rows per request for a number that never changes on a
    given commit). Returns None if the artifact is missing, in which case
    the caller omits those metrics rather than guessing.
    """
    if not FUNNEL_STATS_PATH.exists():
        return None
    return json.loads(FUNNEL_STATS_PATH.read_text())


@st.cache_data(ttl=30)
def count_incidents() -> int | None:
    """Total incidents ever triaged, live from Firestore -- a count()
    aggregation query, not len(load_incidents()) (which is capped at
    INCIDENT_LIMIT and would undercount once more than 50 incidents exist).
    """
    try:
        db = get_client()
        result = db.collection("incidents").count().get()
        return int(result[0][0].value)
    except Exception:
        return None


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
        .aksha-funnel-num {
            font-size: 2.0em; font-weight: 700; color: #E8E8E8; line-height: 1.1;
        }
        .aksha-funnel-label {
            color: #8A8A8A; font-size: 0.85em; margin-top: 0.15em;
        }
        .aksha-trace-card {
            background-color: #161A22; border: 1px solid #2A2E37; border-left: 3px solid #4FA3FF;
            border-radius: 6px; padding: 0.6em 0.9em; margin-bottom: 0.5em;
        }
        .aksha-trace-title { font-weight: 700; font-size: 1.0em; }
        .aksha-trace-model {
            display: inline-block; background-color: #2A3F55; color: #9CC7F0; border-radius: 999px;
            padding: 0.05em 0.55em; font-size: 0.75em; margin-left: 0.5em; font-weight: 600;
        }
        .aksha-trace-what { color: #C7C7C7; font-size: 0.9em; margin-top: 0.2em; }
        .aksha-trace-elapsed { color: #8A8A8A; font-size: 0.8em; margin-top: 0.2em; }
        .aksha-trace-detail { color: #E8E8E8; font-size: 0.88em; margin-top: 0.35em; }
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


# --- Funnel strip -------------------------------------------------------

def render_funnel_strip() -> None:
    """Four numbers, each read from a real artifact -- never hardcoded. The
    first three are precomputed static facts about the offline build
    (scripts/build_funnel_stats.py); the fourth is a live Firestore count.
    Any number whose source is unavailable is omitted, not estimated.
    """
    stats = load_funnel_stats()
    incidents_total = count_incidents()

    cols = st.columns(4)
    cells = [
        ("Windows built", stats.get("windows_built") if stats else None,
         stats.get("windows_built_note", "") if stats else ""),
        ("Windows in test split", stats.get("windows_test") if stats else None,
         stats.get("windows_test_note", "") if stats else ""),
        ("Windows flagged", stats.get("windows_flagged") if stats else None,
         stats.get("windows_flagged_note", "") if stats else ""),
        ("Incidents triaged", incidents_total,
         "live count() on Firestore incidents/"),
    ]
    for col, (label, value, source) in zip(cols, cells):
        with col:
            if value is None:
                st.markdown(
                    f'<div class="aksha-funnel-num" style="color:#555;">n/a</div>'
                    f'<div class="aksha-funnel-label">{label} -- source unavailable, omitted</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="aksha-funnel-num">{value:,}</div>'
                    f'<div class="aksha-funnel-label">{label}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(source)

    st.caption(
        "Each stage is a subset of the one before it. The detector scores "
        "every window and publishes only those above the calibrated "
        "conformal threshold."
    )


# --- Pane 1: Signal ---------------------------------------------------------

def render_signal_pane(incident: dict) -> None:
    channel = incident.get("channel_id", "?")
    t_start, t_end = incident.get("t_start", "?"), incident.get("t_end", "?")
    st.markdown(f'<div class="aksha-pane-title">Signal: {channel}</div>', unsafe_allow_html=True)
    st.caption(f"{t_start} → {t_end} UTC")

    cache = load_signal_cache()
    incident_id = incident["_id"]
    sub = cache[cache["incident_id"] == incident_id] if cache is not None else pd.DataFrame()

    # A dedicated container, rendered into fresh on every call, so a switch
    # between the fallback message and the plot (different element types at
    # the same position) is always a hard replace. Without this, a fragment
    # rerun (the 30s timer or a Load click) can leave the previous incident's
    # chart or fallback showing under the new one until a full page reload.
    pane = st.empty()

    if sub.empty:
        with pane.container():
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
    with pane.container():
        st.pyplot(fig, clear_figure=True)
        st.caption(
            f"Bold blue: {channel} (flagged). Faint gray: the other "
            f"{others['channel_id'].nunique() if not others.empty else 0} lightweight channels, for context. "
            "Shaded band: the incident window."
        )
    plt.close(fig)


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


# --- Agent trace panel --------------------------------------------------

def _fmt_elapsed(prev_ts: str | None, ts: str | None) -> str | None:
    if not prev_ts or not ts:
        return None
    try:
        prev = pd.Timestamp(prev_ts)
        cur = pd.Timestamp(ts)
        return f"+{(cur - prev).total_seconds():.2f}s"
    except (ValueError, TypeError):
        return None


def render_trace_card(step: dict, elapsed: str | None) -> None:
    node = step.get("node", "?")
    label = NODE_LABELS.get(node, {"title": node, "model": False, "what": ""})
    model_badge = '<span class="aksha-trace-model">model call</span>' if label["model"] else ""

    detail_lines: list[str] = []
    if node == "prepare_context":
        detail_lines.append(
            f"reference_windows={step.get('reference_windows')} &middot; "
            f"nearest_labeled={step.get('nearest_labeled')} &middot; "
            f"conformal_p={step.get('conformal_p')}"
        )
    elif node == "assemble_explainer_input":
        refs = ", ".join(step.get("evidence_refs") or []) or "none"
        detail_lines.append(
            f"<b>hypothesis:</b> {step.get('hypothesis')} &middot; "
            f"<b>implicated channel:</b> {step.get('implicated_channel')} &middot; "
            f"<b>confidence:</b> {step.get('investigator_confidence')}"
        )
        detail_lines.append(f"<b>evidence cited:</b> {refs}")
        detail_lines.append(
            f"recognition_distance={step.get('recognition_distance')} "
            f"(percentile {step.get('recognition_percentile')})"
        )
    elif node == "verification_gate":
        detail_lines.append(
            f"<b>gate:</b> {step.get('gate_verdict')} at distance {step.get('gate_distance')} "
            f"vs. threshold {step.get('gate_threshold')}, band [{step.get('band_low')}, {step.get('band_high')}]"
        )
        detail_lines.append(f"<b>model verdict (audit only):</b> {step.get('llm_verdict')}")
        if step.get("llm_reason"):
            detail_lines.append(f"<b>explainer reason:</b> {step['llm_reason']}")
    elif node == "route_by_status":
        detail_lines.append(f"route={step.get('route')} &middot; routing_anomaly={step.get('routing_anomaly')}")
    elif node.startswith("file_report"):
        detail_lines.append(f"severity={step.get('severity')} &middot; log_only={step.get('log_only')}")
    else:
        detail_lines.append(
            f"delivered={step.get('delivered')} &middot; outcome={step.get('routing_outcome')} "
            f"&middot; destination={step.get('routing_destination')}"
        )

    elapsed_html = f'<div class="aksha-trace-elapsed">{elapsed} since previous step</div>' if elapsed else ""
    st.markdown(
        f'<div class="aksha-trace-card">'
        f'<span class="aksha-trace-title">{label["title"]}</span>{model_badge}'
        f'<div class="aksha-trace-what">{label["what"]}</div>'
        f'<div class="aksha-trace-detail">{"<br>".join(detail_lines)}</div>'
        f'{elapsed_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_trace_panel(incident: dict) -> None:
    st.markdown('<div class="aksha-pane-title">Agent trace</div>', unsafe_allow_html=True)

    steps = load_trace(incident["_id"])
    if not steps:
        st.caption("No trace steps found for this incident.")
        return

    prev_ts = None
    for step in steps:
        ts = step.get("timestamp_utc")
        render_trace_card(step, _fmt_elapsed(prev_ts, ts))
        prev_ts = ts

    usage = incident.get("token_usage") or {}
    total_tokens = usage.get("total_tokens")
    thought_tokens = usage.get("thought_tokens")
    triage_seconds = incident.get("triage_seconds")
    llm_calls = incident.get("llm_calls")
    cost_bits = []
    if total_tokens is not None:
        cost_bits.append(f"{total_tokens:,} tokens" + (f" ({thought_tokens:,} thinking)" if thought_tokens else ""))
    if llm_calls is not None:
        cost_bits.append(f"{llm_calls} model calls")
    if triage_seconds is not None:
        cost_bits.append(f"{triage_seconds:.2f}s wall time")
    if cost_bits:
        st.caption("Cost and latency (stored on the incident doc): " + " · ".join(cost_bits))


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

    st.caption(
        "Replay publishes the Mission2 test split in chronological order. "
        "Only windows the live detector scores above the conformal threshold "
        "become incidents."
    )


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

    render_funnel_strip()
    st.divider()

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
    if selected:
        render_trace_panel(selected)
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
