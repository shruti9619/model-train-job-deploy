# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copy code
COPY . .

# Install package
RUN pip install --no-cache-dir .

# Default command
CMD ["python", "-m", "classification_pipeline.run", "--config-path", "dtree_config.yaml"]