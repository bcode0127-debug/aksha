# AKSHA -- PRD v0.2
*Aug 11, 2026. Status: locked (design phase complete).*

**1. Problem**

Satellite telemetry triage is still largely manual. Mission control systems perform limit checking, expected-status checks, and derived-parameter computation, with complex spacecraft carrying 6000+ monitored parameters (ESA Bulletin 89). Detection in practice is mostly fixed-threshold out-of-limit alarms, which are context-blind, and at telemetry volume even a small false-alarm rate floods the control room until operators ignore alerts (ML-for-reliability review, arXiv 2008.08221).

A detector emits a score, not a decision. It doesn't say which channel drifted, whether the pattern matches a known failure mode, whether it's real or noise, or who should be woken up. The gap between anomaly score and actionable incident is closed by a human today, and under distribution shift the score is least trustworthy exactly when it matters most.

**2. Person**

Three recipient tiers, each with a different threshold for interruption:
- Console operator: on shift, sees everything
- Subsystem engineer: paged into their domain
- Flight director: escalation for mission-impacting decisions

**3. What AKSHA does**

Automates the operator triage loop: open the trend, pull correlated context, check history, decide real vs false alarm, assign severity, file, escalate or log. Event-driven, runs in the background on a telemetry stream, no human driving steps.

**4. Scope**

In: 5-stage graph (2 LLM nodes, 3 function nodes) covering detect, investigate, explain, file_report, route; severity-gated routing to 3 tiers; Streamlit dashboard; replay loader; eval harness.
Gated stretch (only if the Aug 16 skeleton lands with room): autoencoder on raw signals as a second detector, fused via AOM/MOA.
Out: cascade prediction, OpenTelemetry, digital twin, ESA-ADB as build data, physics-informed models, Open MCT as a dependency.

**5. Success criteria**

Stage One (pass/fail): Gemini 3.5 via Vertex AI, ADK, GCP services, GCP deployment proof in video, repo, architecture diagram, README spin-up, ~4-min video.
Stage Two: autonomous multi-step completion with no human intervention (40%); decoupling, state management, scoped tools, failure handling (30%); unedited live execution with visible logs and database writes (30%).
Product: end-to-end run on ESA-ADB Mission2 with at least one gate rejection and one correct escalation, measured against ground-truth labels.
Evaluation integrity: report the 7 benchmark-style metrics (Accuracy, Precision, Recall, F1, MCC, AUC_ROC, AUC_PR) on ESA-ADB Mission2's own train/calibration/test split (137,235/23,970/168,410 windows), positive class `rare_event ∪ anomaly` -- Mission2's test split carries only 4 true anomaly-labelled windows from a single event, too few to support an anomaly-only metric on its own (ADR-015; docs/mission2-adapter-notes.md). Point-adjusted F1 is never reported (Kim et al., AAAI 2022; PA originates with Xu et al., WWW 2018).

**6. Constraints**

Solo. Submit Aug 29; hard close Aug 31, 5pm PT. $150 GCP credits; Gemini Flash by default; min instances 0; resources deleted after recording. Mac mini only. ESA-ADB Mission2 lightweight subset, channels 18-28 (CC BY 3.0 IGO, cited; see ADR-015). Fresh repo; all code written in the submission window. PoC bar, not research grade.

**7. Hard architectural rule**

LLMs never receive raw telemetry arrays: float tokenization degrades precision, long windows cause causal misattribution, and models synthesize physically impossible state transitions. Core does all numerics; agents receive structured summaries only.

**8. Prior art and positioning**

No first-of-kind claims. Agentic anomaly triage exists in NASA ground infrastructure (Chou et al., arXiv 2508.21111, JPL DSN). AKSHA's position: domain shift to spacecraft telemetry, a deterministic calibrated gate deciding, with an independent LLM explaining, event-driven rather than batch, public benchmark, fully reproducible. Survey citation: Fejjari et al., MDPI Applied Sciences 2025. No standards claims (CCSDS/ECSS/NASA) anywhere without a document number and link.

**9. Name**

AKSHA (Sanskrit अक्ष): axis, eye, indestructible. One spelling everywhere.
