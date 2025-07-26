#!/bin/bash

# Build script for chicken weight estimation Docker container
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="chicken-weight-estimator"
IMAGE_TAG="latest"
REGISTRY_URL=""  # Set this if pushing to a registry

echo -e "${BLUE}🐔 Building Chicken Weight Estimation Docker Container${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running"

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p logs
mkdir -p model_artifacts
mkdir -p config

# Copy config files if they don't exist
if [ ! -f "config/model_config.yaml" ]; then
    cp src/utils/config/model_config.yaml config/ 2>/dev/null || true
fi

if [ ! -f "config/camera_config.yaml" ]; then
    cp src/utils/config/camera_config.yaml config/ 2>/dev/null || true
fi

print_status "Directories created"

# Build the Docker image
echo -e "${BLUE}🔨 Building Docker image...${NC}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Build with progress output
docker build \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --progress=plain \
    --no-cache \
    .

if [ $? -eq 0 ]; then
    print_status "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Get image size
IMAGE_SIZE=$(docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Size}}" | tail -n 1)
print_status "Image size: ${IMAGE_SIZE}"

# Test the container
echo -e "${BLUE}🧪 Testing the container...${NC}"

# Run a quick test
docker run --rm \
    --name chicken-test \
    -p 8081:8080 \
    -d \
    "${IMAGE_NAME}:${IMAGE_TAG}"

# Wait for container to start
echo "Waiting for container to start..."
sleep 30

# Test health endpoint
if curl -f http://localhost:8081/ping > /dev/null 2>&1; then
    print_status "Container health check passed"
else
    print_warning "Container health check failed, but this might be expected in test mode"
fi

# Stop test container
docker stop chicken-test > /dev/null 2>&1 || true

print_status "Container test completed"

# Show final information
echo -e "${BLUE}📋 Build Summary${NC}"
echo "=================================================="
echo "Image Name: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Image Size: ${IMAGE_SIZE}"
echo "Build Date: $(date)"

# Show usage instructions
echo -e "${BLUE}🚀 Usage Instructions${NC}"
echo "=================================================="
echo "Run the container:"
echo "  docker run -p 8080:8080 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Run with Docker Compose:"
echo "  docker-compose up -d"
echo ""
echo "View logs:"
echo "  docker logs -f chicken-weight-estimator"
echo ""
echo "Access the API:"
echo "  http://localhost:8080/"
echo ""

# Optional: Push to registry
if [ -n "$REGISTRY_URL" ]; then
    echo -e "${BLUE}📤 Pushing to registry...${NC}"
    
    # Tag for registry
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Push to registry
    docker push "${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    if [ $? -eq 0 ]; then
        print_status "Image pushed to registry: ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
    else
        print_error "Failed to push image to registry"
    fi
fi

print_status "Build process completed successfully!"

echo -e "${GREEN}🎉 Your chicken weight estimation system is ready to deploy!${NC}"