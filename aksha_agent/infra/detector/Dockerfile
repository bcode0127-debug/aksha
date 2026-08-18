# Build context is the staging directory produced by scripts/deploy_detector.sh,
# not this directory: the service imports `aksha_core` and loads the committed
# detector artifact, both of which live above this folder in the repo and so
# cannot be COPYed from here directly.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY aksha_core ./aksha_core
COPY detector_service.py .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn detector_service:app --host 0.0.0.0 --port ${PORT:-8080}"]
