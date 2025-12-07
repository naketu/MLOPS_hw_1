FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

COPY ml_api.py .
COPY s3_functions.py .
COPY dashboard.py .

EXPOSE 8000

CMD ["uvicorn", "ml_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
