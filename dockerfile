# Stage - 1 : Builder

FROM python:3.10-slim AS builder

# intall system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirementss first before any source code to leverage docker cache
COPY requirements.txt .

# Buildkit cache mount -pip cache persist between builds on same machine

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# Stage - 2 : Final image
FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only necessary files from builder stage
COPY --from=builder /install /usr/local/
COPY src/ ./src/

COPY params.yaml .

# model/ is intentionally NOT copied 
# It is mounted as a volume in docker-compose.yml
# This means you can update the model without rebuilding the image
# Just replace the .joblib files and restart the container

# Non-root user for security
RUN useradd -m appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app/logs

USER appuser

# Image metadata
LABEL maintainer="Pavan Modi" \
      description="A dummy emotion detection API built with FastAPI" \
      version="1.0.0" \
      license="MIT" 

# port
EXPOSE 8000

# healthcheck to ensure container is healthy before accepting traffic
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

# command to run the app
CMD ["uvicorn", "src.api.app:app", \
     "--host",\
     "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]





