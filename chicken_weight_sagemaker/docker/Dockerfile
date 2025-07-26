# SageMaker-compatible Docker container for chicken weight estimation
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_CONTAINER_LOG_LEVEL=20
ENV SAGEMAKER_REGION=us-east-1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml

# Create necessary directories
RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/input /opt/ml/output

# Copy requirements first for better caching
COPY docker/requirements.txt /opt/ml/code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Install additional dependencies for computer vision
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    filterpy==1.4.5 \
    lap==0.4.0 \
    psutil==5.9.0

# Copy source code
COPY src/ /opt/ml/code/src/
COPY docker/inference.py /opt/ml/code/inference.py

# Copy model artifacts (will be overridden by SageMaker)
COPY model_artifacts/ /opt/ml/model/

# Set permissions
RUN chmod +x /opt/ml/code/inference.py

# Expose port for SageMaker
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Set the entry point for SageMaker
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]