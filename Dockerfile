# Image for the forecast API (default command) and the pipeline.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# libgomp is needed by xgboost wheels; make is used by the pipeline targets
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 make curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock ./
RUN pip install -r requirements.lock

COPY pyproject.toml README.md Makefile ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY examples ./examples
RUN pip install --no-deps -e . \
    && mkdir -p results data \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "voltail.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
