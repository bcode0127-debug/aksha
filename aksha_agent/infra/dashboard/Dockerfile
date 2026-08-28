# Build context is the staging directory produced by scripts/deploy_dashboard.sh,
# not this directory: the dashboard reads the committed signal cache and funnel
# stats artifacts, which live above this folder in the repo, plus the dark-theme
# config, none of which can be COPYed from here directly.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Read-only content the dashboard needs at runtime. No aksha_core or
# aksha_agent.graph import at all -- app.py only reads Firestore and these
# three local files (the third is optional, the app falls back if absent).
COPY aksha_agent/dashboard ./aksha_agent/dashboard
COPY .streamlit ./.streamlit
COPY data/processed/dashboard_signal_cache.parquet ./data/processed/dashboard_signal_cache.parquet
COPY aksha_core/artifacts/mission2_funnel_stats.json ./aksha_core/artifacts/mission2_funnel_stats.json

EXPOSE 8080

CMD ["sh", "-c", "streamlit run aksha_agent/dashboard/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]
