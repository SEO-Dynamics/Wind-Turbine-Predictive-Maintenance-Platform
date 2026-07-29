# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Single image serving both the API and the dashboard. Which one runs is chosen
# by the compose command, so both services share one build and one artifact set.
#
# The Python version and the pinned requirements deliberately match the
# environment that trains the model. A serialised scikit-learn estimator is only
# guaranteed to load under the version that wrote it, and compose mounts the
# host's artifacts/ into the container - so a mismatched image would unpickle a
# model built by a different scikit-learn.
#
# The image never trains a model. Artifacts are mounted in at runtime (see
# docker-compose.yml); a container started without them still boots and reports
# `degraded` on /health with the command needed to build them.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WTPM_PROJECT_ROOT=/app

WORKDIR /app

# Build tooling needed by numpy/scipy/shap wheels on slim, plus curl for the
# container health checks.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so application edits do not invalidate the wheel cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/

RUN pip install --no-cache-dir --no-deps -e .

# Artifact and data directories exist even when nothing is mounted, so the
# services degrade gracefully instead of crashing on a missing path.
RUN mkdir -p data/raw data/interim data/processed data/samples \
             artifacts/models artifacts/metrics artifacts/figures artifacts/metadata

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default target is the API; compose overrides this for the dashboard.
CMD ["uvicorn", "wind_turbine_pm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
