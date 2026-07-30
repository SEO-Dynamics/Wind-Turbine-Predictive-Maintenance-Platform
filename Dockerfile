# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# One hardened image serves the API, dashboard, and opt-in training pipeline.
#
# Native build tools live only in the builder stage. The final runtime contains
# the hashed runtime lock, application wheel, and no compiler, curl, or test
# tooling. Model artifacts and generated data remain external volume mounts.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# A source build is only a fallback when a dependency has no compatible wheel.
# None of these packages are copied into the final runtime image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip wheel \
    --require-hashes \
    --wheel-dir /wheels \
    --requirement requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip wheel --no-deps --wheel-dir /wheels .


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WTPM_PROJECT_ROOT=/app

WORKDIR /app

COPY requirements.txt ./
COPY --from=builder /wheels /wheels
RUN python -m pip install \
        --no-index \
        --find-links=/wheels \
        --require-hashes \
        --requirement requirements.txt \
    && python -m pip install \
        --no-index \
        --find-links=/wheels \
        --no-deps \
        wind-turbine-pm \
    && rm -rf /wheels

# Runtime application files. The Python package itself is installed from the
# wheel built above; source and test tooling are not copied into this stage.
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/

# Services only read these paths. The opt-in pipeline replaces them with
# writable host mounts when it is explicitly invoked.
RUN mkdir -p data/raw data/interim data/processed data/samples \
             artifacts/models artifacts/metrics artifacts/figures artifacts/metadata \
    && groupadd --gid 1000 appuser \
    && useradd --create-home --uid 1000 --gid 1000 appuser \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000 8501

# Compose supplies service-specific checks. This default protects direct
# `docker run` usage without adding curl to the runtime image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

CMD ["uvicorn", "wind_turbine_pm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
