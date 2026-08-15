FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY triage_service.py .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn triage_service:app --host 0.0.0.0 --port ${PORT:-8080}"]
