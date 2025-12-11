# MaxQ Backend Dockerfile
# Python 3.11 slim for optimal size and compatibility

FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY setup.py .
COPY maxq/ ./maxq/

# Install dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn fastapi

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create directories for data persistence
RUN mkdir -p /root/.maxq /root/.cache/fastembed

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV QDRANT_URL=http://qdrant:6333

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "maxq.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
