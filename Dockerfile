# Build: docker build -t wafer-api .
# Run:  docker compose up
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY wafer_sim ./wafer_sim
COPY ml ./ml
COPY api ./api

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[api,ml]"

ENV DATA_DIR=/app/data
ENV ML_ARTIFACTS_DIR=/app/ml/artifacts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
