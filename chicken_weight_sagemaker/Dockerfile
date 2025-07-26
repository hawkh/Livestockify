# Multi-stage Docker build for chicken weight estimation system
# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV FLASK_APP=demo_server.py
ENV FLASK_ENV=production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/src /app/models /app/logs /app/config && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser demo_server.py /app/
COPY --chown=appuser:appuser simple_test.py /app/
COPY --chown=appuser:appuser test_functionality_simple.py /app/

# Copy configuration files
COPY --chown=appuser:appuser src/utils/config/*.yaml /app/config/

# Create model artifacts directory (for future model files)
RUN mkdir -p /app/model_artifacts && \
    chown -R appuser:appuser /app/model_artifacts

# Create startup script
COPY --chown=appuser:appuser <<EOF /app/start.sh
#!/bin/bash
set -e

echo "🐔 Starting Chicken Weight Estimation System..."
echo "📊 Running system tests..."

# Run basic functionality test
python test_functionality_simple.py

if [ \$? -eq 0 ]; then
    echo "✅ System tests passed!"
    echo "🚀 Starting web server..."
    exec python demo_server.py
else
    echo "❌ System tests failed!"
    exit 1
fi
EOF

RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Set the entry point
ENTRYPOINT ["/app/start.sh"]