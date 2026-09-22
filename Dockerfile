FROM python:3.11-slim

WORKDIR /app

# Configure pip timeout and retries for network resilience
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python package definition and install RepoMind dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -e .

# Copy application code
COPY repomind/ repomind/

# Create data directory
RUN mkdir -p /app/.repomind_data

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["python", "-m", "repomind.api.main"]

