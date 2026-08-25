# AKSHA

Anomaly detection for spacecraft telemetry, with conformal-calibrated confidence and an LLM agent in the loop for triage and explanation.

## Architecture

![AKSHA architecture](docs/architecture.svg)

Generated straight from the live `Workflow.graph.edges` object (`scripts/render_diagram.py`), not hand-drawn — rename a node and this changes with it (ADR-013). Regenerate with:
```
python3 scripts/render_diagram.py
```

Design docs: [PRD](docs/PRD.md), [TRD](docs/TRD.md), [ADRs](docs/adr/).

## Spin-up

The trained detector, calibration, and context reference are committed
(`aksha_core/artifacts/`), so running the deployed pipeline needs neither
fetch step below. They're only required to retrain or recalibrate from raw
data (`scripts/train_detector.py`, `scripts/calibrate_recognition.py`,
`scripts/build_context_reference.py`).

1. `python3 scripts/fetch_mission2.py` — downloads and unpacks ESA-ADB's
   Mission2 (3.8 GB, CC BY 3.0 IGO, the build dataset — see
   [ADR-015](docs/adr/ADR-015.md)) into `data/`, checksum-verified.
2. `python3 scripts/fetch_data.py` — downloads `dataset.csv` and
   `segments.csv` (OPSSAT-AD, cited for operator-requirements framing only —
   see [ADR-015](docs/adr/ADR-015.md)) into `data/`, checksum-verified.
3. Enable the required GCP APIs:
   ```
   gcloud services enable run.googleapis.com pubsub.googleapis.com firestore.googleapis.com aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com billingbudgets.googleapis.com
   ```

_Remaining steps: placeholder._

## Dashboard

Read-only view over live Firestore state — recent incidents, per-node trace,
and the gate-vs-LLM audit block for whichever one you select. No auth, no
writes. Run it locally with ADC (`gcloud auth application-default login`)
against the same project the services use:
```
streamlit run aksha_agent/dashboard/app.py
```

## Evaluation

_Placeholder._

## Disclosure

All code written during the All Things Agentic submission period. No pre-existing code incorporated.

## Data & Citation

The detector, gate calibration, and golden set are built on **ESA-ADB's Mission2** benchmark (CC BY 3.0 IGO), specifically its lightweight channel subset (channels 18–28) — see [ADR-015](docs/adr/ADR-015.md):

> Kotowski, K., Haskamp, C., Andrzejewski, J., Ruszczak, B., Nalepa, J., Lakey, D., Collins, P., Kolmas, A., Bartesaghi, M., Martínez-Heras, J., De Canio, G. (2024). *European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry*. arXiv:2406.17826. Dataset: Zenodo, DOI: [10.5281/zenodo.15237121](https://doi.org/10.5281/zenodo.15237121).

**OPSSAT-AD** (CC-BY 4.0) is cited for its operator-requirements framing, not as build data — see [ADR-006](docs/adr/ADR-006.md) (superseded by ADR-015):

> Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. et al. (2025). *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*. Scientific Data. Dataset: Zenodo, DOI: [10.5281/zenodo.15108715](https://doi.org/10.5281/zenodo.15108715).

## License

Apache License 2.0. See [LICENSE](LICENSE).
