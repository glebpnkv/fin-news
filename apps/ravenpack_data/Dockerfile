ARG PYTHON_VERSION=3.12.12
FROM python:${PYTHON_VERSION}-slim

# Set the working directory in the container
WORKDIR /app

# System build deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    bash \
  && rm -rf /var/lib/apt/lists/*

# Install uv and sync project deps into a venv at /app/.venv
# Copy app code
COPY . .
# COPY pyproject.toml .
RUN python -m pip install -U pip && \
    python -m pip install --user uv
# Make `finnews` (and other top-level packages under /app) importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENV PATH="/root/.local/bin:${PATH}"
# Installing dependencies for the root fin-news and fin-news-ravenpack-data packages
RUN uv sync --package fin-news --package fin-news-ravenpack-data
ENV PATH="/app/.venv/bin:${PATH}"

# Ensure pip is available in the venv (for Vertex components which sometimes invoke it)
RUN python -m ensurepip || true && python -m pip install -U pip

# Removing build deps to slim the final image
RUN apt-get purge -y --auto-remove build-essential python3-dev git libffi-dev || true && \
    rm -rf /var/lib/apt/lists/*

# Copy app code
# COPY . .

# Install the local package
WORKDIR /app/apps/ravenpack_data
RUN python -m pip install -e .
