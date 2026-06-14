FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

COPY src ./src
COPY config ./config
RUN pip install -e .

RUN mkdir -p /app/data/postings /app/data/extractions \
             /app/data/metrics /app/data/reports

ENV SKILLTREND_DATA_DIR=/app/data

EXPOSE 8501

# Default: interactive shell so you can pick scan/extract/tui/web at runtime.
# Override with `docker compose run web` etc.
CMD ["bash"]
