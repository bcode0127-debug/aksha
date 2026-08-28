# AKSHA -- TRD v0.4
*Aug 11, 2026. Status: locked.*

**1. Pipeline (two-topic split-queue)**

Amended by [ADR-016](adr/ADR-016.md): the publish step from detector service
to the `triage` topic is conditional on the conformal threshold (is_anomalous),
not unconditional as this diagram originally specified. The Firestore write
of the DetectionResult happens for every scored window regardless.

```
Mission2 fragments (ESA-ADB; see ADR-015)
  → replay loader (publishes in sequence)
  → Pub/Sub topic: telemetry-in  [push]
  → Cloud Run: detector service (fast path, ms-scale)
      detect (IForest + split conformal, deterministic)
      → persist DetectionResult (Firestore), every scored window
      → ack immediately
      → [conditional: is_anomalous, i.e. score >= threshold]
      → publish to Pub/Sub topic: triage  [push, ack deadline 600s]
  → Cloud Run: triage service (slow path, ADK graph workflow)
      investigate → assemble_explainer_input → explain → verification_gate
        → [router: gate verdict] → file_report
        → [router: severity] → route
  → Firestore (incidents + traces)
  → Streamlit dashboard (reads Firestore)
  → 3 Slack webhooks, severity-gated
```

LLM latency never gates a broker ack. Both subscriptions carry dead-letter topics (5 delivery attempts). All workers idempotent under at-least-once delivery. Flip threshold: if per-incident LLM latency approaches 600s, move the triage leg to pull with lease extension or Cloud Tasks (ADR-011).

Measured: ADK 2 spike single-LLM-call wall time was 14.7–38.8s (tutorial marathon-strategy workflow, AI Studio backend, Aug 13 2026). A two-call incident (investigate + explain) sits far under the 600s ack ceiling -- the flip threshold above stays theoretical for now.

**2. Module boundary**

`aksha_core/`: pure Python, PyTorch, scikit-learn, PyOD. Zero `google.*`, ADK, or vertexai imports. CI grep enforces.
`aksha_agent/`: ADK agents, model clients, GCP infra, dashboard. Imports core; never the reverse.

**3. Boundary contract**

The only object crossing into agent land:

```
DetectionResult:
  fragment_id, channel_id, t_start, t_end
  score, threshold, conformal_p
  features: dict[str, float]        # the 18 benchmark features
  detector_version
```

No raw arrays cross. Investigate additionally receives a summarized channel history and k nearest labeled fragments as feature summaries.

**4. Detection (restructured per ADR-012)**

Primary: Isolation Forest on the 18 precomputed features from `dataset.csv`, exactly the benchmark protocol (PyOD, contamination = 0.2), plus split conformal calibration on its scores using the published train split as proper training + calibration (inductive conformal anomaly detection, Laxhammar & Falkman 2015). Output: score, threshold, conformal_p.
Gated stretch: autoencoder on raw signals from `segments.csv` (genuinely different representation), trimmed training (Zhou & Paffenroth, KDD 2017), fused via AOM/MOA with Z-score standardization (Aggarwal & Sathe; PyOD). Gate: Aug 16 skeleton lands with room. Convention-not-sourced where kept: interpolation choices, AE layer sizing, robust-loss choice.

**5. Graph nodes (per ADR-013)**

Five stages: 2 agent nodes, 3 function nodes. Every boundary validated by `input_schema`.

- **detect** -- function node, 0 LLM. IForest + split conformal (section 4)
- **investigate** -- agent node, Gemini 3.5 Flash, `mode="single_turn"`. Outputs hypothesis, implicated channel, evidence refs, confidence
- **assemble_explainer_input** -- function node, 0 LLM. Builds the explainer's input as DetectionResult + evidence + hypothesis, dropping investigator_confidence. This is the mechanism that enforces ADR-005: isolation is an edge in the graph, visible in `Workflow.graph.edges`
- **explain** -- agent node, `mode="single_turn"`, separate call, `temperature=0.0`. Outputs the operator-facing `reason` plus an independent `status` that is RECORDED FOR AUDIT ONLY. It cannot change the verdict
- **verification_gate** -- function node, 0 LLM. The decision. Compares the calibrated distance against bounds loaded from the calibration artifact: `>= band.high` → confirm, `<= band.low` → reject, otherwise → disputed
- **file_report** -- function node, 0 LLM. Deterministic assembly, computes severity
- **route** -- dict-edge router on severity, writes routing outcome back

Two dict-edge routers: gate verdict (reject and disputed → log, never escalate) and severity (3 destinations). Both carry DEFAULT_ROUTE (section 9). Two LLM calls per incident, total. Node names are part of the routing contract; any rename greps all docs same day.

**6. Incident document**

```
incident_id, timestamp_utc
channel_id, fragment_id, detector_version
anomaly_type          # point | contextual | collective | correlation (SFAD taxonomy)
anomaly_score, threshold, conformal_p
features_summary
investigator_hypothesis, investigator_confidence
gate_verdict, final_verdict, gate_distance, gate_threshold, band_low, band_high, llm_verdict (audit only), llm_reason
severity              # Critical | Caution | Advisory
routing_destination, routing_outcome, routing_timestamp_utc
status                # open | in_progress | closed
reasoning_trace
```

No ground-truth labels in runtime documents; labels exist only in the eval harness. No standards claimed.

**7. Severity and routing**

Critical → flight director. Caution → subsystem engineer. Advisory → log only. Computed deterministically from conformal_p, the gate verdict, and channel criticality (our mapping, disclosed as ours). `disputed` never escalates.

**8. State**

Firestore: `incidents/{id}`, `traces/{incident_id}/steps/{n}`, `runs/{run_id}`. Every agent step appends a trace doc: audit trail and dashboard drill-down in one structure. No in-process session state; both services scale to zero. Idempotency key `fragment_id + detector_version` checked before any write.

No node-level callbacks exist in ADK (ADR-013 spike) -- model-level plugin hooks (`before_model_callback`/`after_model_callback`) cover the two LLM nodes' side of tracing; function nodes append their trace docs explicitly in code, not via a framework hook.

**9. Failure handling**

- **Unmatched route** → both route dicts carry a DEFAULT_ROUTE entry to the log tier with the incident flagged as a routing anomaly. Invariant: no incident leaves the graph unrecorded. Without this, an unmatched route ends the branch silently and the process exits 0 with no output (confirmed empirically, ADR-013 spike)
- **Malformed LLM output** → `input_schema` validates a node's *input*, not the output of the node that produced it -- so a bad LLM output doesn't fail where it was generated, it surfaces as a `pydantic.ValidationError` at the *next* node's input gate (ADR-013 spike). Node-level retry cannot repair this: it's a deterministic failure, not a transient one (ADR-014). Policy: fail fast, no repair loop, deterministic fallback marked `llm_unavailable`
- **Retry policy** (ADR-014) → `retry_config` enabled only on the two LLM nodes (investigate, explain), `exceptions` allowlist limited to transient transport failures and `NodeTimeoutError`. Function nodes carry no `retry_config` -- a failure there is a bug, not a transient condition
- Calibrated distance inside the ambiguous band → `disputed`, log tier, no escalation. Gate/explainer disagreement is recorded for audit and does NOT affect the outcome
- Vertex timeout or quota → per-node `timeout` raises `NodeTimeoutError`, which is retry-compatible (ADR-014); exponential backoff, then dead-letter once retries exhaust
- Pub/Sub redelivery → idempotency key check
- Runaway agent → hard max-turn cap per incident

**10. Eval harness (offline, never imported by runtime)**

Reads `dataset.csv` labels and the official split. Produces: the 7 benchmark metrics for our detector beside the 30 published baselines; gate rejection accuracy (rejections vs ground truth); the dashboard evidence panel. This is the demo's proof segment.

**11. Security**

Per-tool least privilege (detect can't write Firestore, file_report can't route, route can't re-detect). Secrets in Secret Manager. Dedicated service account, least-privilege IAM. Slack webhooks are secrets. Cloud Run requires auth.

**12. Cost**

Flash for investigate, small Gemma for explain, Pro never. Min instances 0, max 2. Budget alert on. Record GCP proof, then delete resources.

**13. Open**

Gemma hosting: local at demo vs Vertex endpoint.
