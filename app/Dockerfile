### Human Promoter Atlas — Streamlit container ###############################
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/app

# System deps: curl for HEALTHCHECK; tini as PID 1 for clean signal handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl tini && \
    rm -rf /var/lib/apt/lists/*

# Pin runtime deps explicitly here (kept small; build-time-only deps excluded)
COPY app/requirements.txt /srv/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and lib code
COPY app/ /srv/app/app/

# Streamlit reads .streamlit/config.toml from the working directory.
COPY .streamlit/ /srv/app/.streamlit/

# The DuckDB + aggregate parquet are NOT baked into the image — they ride in
# via the docker-compose volume mount (./data:/srv/app/data:ro). Keeps the
# image lean (~700 MB) and lets us swap the data layer without rebuilding.
RUN mkdir -p /srv/app/data

# Non-root user
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /srv/app
USER appuser

EXPOSE 8501

# Single-process Streamlit; Caddy handles TLS + buffering in front
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
