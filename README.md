# AKSHA

Anomaly detection for spacecraft telemetry, with conformal-calibrated confidence and an LLM agent in the loop for triage and explanation.

## Architecture

_Placeholder: diagram generated from `Workflow.graph.edges`._

Design docs: [PRD](docs/PRD.md), [TRD](docs/TRD.md), [ADRs](docs/adr/).

## Spin-up

1. `python3 scripts/fetch_data.py` — downloads `dataset.csv` and `segments.csv` (OPSSAT-AD) into `data/`, checksum-verified.

_Remaining steps: placeholder._

## Evaluation

_Placeholder._

## Disclosure

All code written during the All Things Agentic submission period. No pre-existing code incorporated.

## Data & Citation

This project uses the **OPSSAT-AD** dataset (CC-BY 4.0):

> Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. et al. (2025). *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*. Scientific Data. Dataset: Zenodo, DOI: [10.5281/zenodo.15108715](https://doi.org/10.5281/zenodo.15108715).


Operator requirements are sourced from **ESA-ADB** (European Space Agency Anomaly Detection Benchmark).

## License

Apache License 2.0. See [LICENSE](LICENSE).
