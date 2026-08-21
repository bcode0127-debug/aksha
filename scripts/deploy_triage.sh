#!/usr/bin/env bash
# Deploy the triage service (the ADK graph with two real Gemini agent nodes).
#
# Stages exactly what the image needs, for the same reason as the detector:
# `gcloud run deploy --source` builds from one directory and cannot reach above
# it, but the service imports `aksha_agent.graph` and reads the committed
# context reference from `aksha_core/artifacts/`.
#
# GOOGLE_CLOUD_LOCATION is `global`, NOT the Cloud Run region. Verified against
# Vertex in this project: gemini-3.5-flash and gemini-3.5-flash-lite exist only
# on the global endpoint and 404 on every regional one. The two settings are
# independent — the container still runs in us-central1.
#
# Usage:  scripts/deploy_triage.sh [PROJECT_ID] [RUN_REGION]
set -euo pipefail

PROJECT="${1:-aksha-hackathon}"
RUN_REGION="${2:-us-central1}"
SERVICE="triage-service"
RUNTIME_SA="aksha-runtime@${PROJECT}.iam.gserviceaccount.com"

MODEL_LOCATION="global"
INVESTIGATOR_MODEL="${AKSHA_INVESTIGATOR_MODEL:-gemini-3.5-flash}"
VERIFIER_MODEL="${AKSHA_VERIFIER_MODEL:-gemini-3.5-flash-lite}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/aksha_agent/infra/triage"
REFERENCE="${REPO_ROOT}/aksha_core/artifacts/mission2_context_reference.json"

if [[ ! -f "${REFERENCE}" ]]; then
  echo "context reference missing: ${REFERENCE}" >&2
  echo "build it first:  python3 scripts/build_context_reference.py" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

cp "${SRC}/Dockerfile" "${SRC}/requirements.txt" "${SRC}/triage_service.py" "${STAGE}/"

# aksha_agent.graph (the workflow) and aksha_core/artifacts (the context
# reference). The detector's model artifact is NOT shipped here: this service
# never scores anything.
( cd "${REPO_ROOT}" && \
  find aksha_agent/graph aksha_agent/__init__.py -type f -name '*.py' \
    -not -path '*/__pycache__/*' -print0 \
  | while IFS= read -r -d '' f; do
      mkdir -p "${STAGE}/$(dirname "$f")"; cp "$f" "${STAGE}/$f"
    done )
mkdir -p "${STAGE}/aksha_core/artifacts"
touch "${STAGE}/aksha_core/__init__.py"
cp "${REFERENCE}" "${STAGE}/aksha_core/artifacts/"

echo "staged for build:"
( cd "${STAGE}" && find . -type f | sort | sed 's/^/  /' )
echo

gcloud run deploy "${SERVICE}" \
  --source "${STAGE}" \
  --project "${PROJECT}" \
  --region "${RUN_REGION}" \
  --no-allow-unauthenticated \
  --service-account "${RUNTIME_SA}" \
  --min-instances 0 \
  --max-instances 2 \
  --memory 1Gi \
  --timeout 600 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=${MODEL_LOCATION},GOOGLE_GENAI_USE_ENTERPRISE=True,AKSHA_INVESTIGATOR_MODEL=${INVESTIGATOR_MODEL},AKSHA_VERIFIER_MODEL=${VERIFIER_MODEL}" \
  --quiet
