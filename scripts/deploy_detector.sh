#!/usr/bin/env bash
# Deploy the detector service to Cloud Run.
#
# `gcloud run deploy --source` builds from a single directory and expects the
# Dockerfile at its root, but the service imports `aksha_core` and loads the
# committed detector artifact, both of which live above the service folder.
# So this stages exactly what the image needs into a temp directory and builds
# from there — explicit about what ships, rather than sending the whole repo.
#
# Usage:  scripts/deploy_detector.sh [PROJECT_ID] [REGION]
set -euo pipefail

PROJECT="${1:-aksha-hackathon}"
REGION="${2:-us-central1}"
SERVICE="detector-service"
RUNTIME_SA="aksha-runtime@${PROJECT}.iam.gserviceaccount.com"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/aksha_agent/infra/detector"
ARTIFACT="${REPO_ROOT}/aksha_core/artifacts/mission2_iforest.joblib"

if [[ ! -f "${ARTIFACT}" ]]; then
  echo "detector artifact missing: ${ARTIFACT}" >&2
  echo "train it first:  python3 scripts/train_detector.py" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

cp "${SRC}/Dockerfile" "${SRC}/requirements.txt" "${SRC}/detector_service.py" "${STAGE}/"

# aksha_core, minus caches and anything the service does not need at runtime.
mkdir -p "${STAGE}/aksha_core"
( cd "${REPO_ROOT}" && \
  find aksha_core -type f \( -name '*.py' -o -name '*.joblib' -o -name '*.json' \) \
    -not -path '*/__pycache__/*' -print0 \
  | while IFS= read -r -d '' f; do
      mkdir -p "${STAGE}/$(dirname "$f")"
      cp "$f" "${STAGE}/$f"
    done )

echo "staged for build:"
( cd "${STAGE}" && find . -type f | sort | sed 's/^/  /' )
echo

gcloud run deploy "${SERVICE}" \
  --source "${STAGE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --no-allow-unauthenticated \
  --service-account "${RUNTIME_SA}" \
  --min-instances 0 \
  --max-instances 2 \
  --memory 512Mi \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=${REGION}" \
  --quiet
