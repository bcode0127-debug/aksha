# Build context is the staging directory produced by scripts/deploy_triage.sh,
# not this directory: the service imports `aksha_agent.graph` and loads the
# committed context reference from `aksha_core/artifacts/`, both of which live
# above this folder in the repo.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY aksha_agent ./aksha_agent
COPY aksha_core ./aksha_core
COPY triage_service.py .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn triage_service:app --host 0.0.0.0 --port ${PORT:-8080}"]
