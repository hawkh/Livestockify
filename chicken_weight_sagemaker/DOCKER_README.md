# 🐔 Chicken Weight Estimation System - Docker Deployment

This guide covers containerizing and deploying the chicken weight estimation system using Docker.

## 📋 Prerequisites

- Docker Desktop installed and running
- At least 4GB RAM available for containers
- PowerShell (Windows) or Bash (Linux/Mac)

## 🚀 Quick Start

### Option 1: Using PowerShell Scripts (Recommended for Windows)

```powershell
# Build the Docker image
.\build-docker.ps1

# Run the container
.\run-docker.ps1 -Detached
```

### Option 2: Using Docker Commands Directly

```bash
# Build the image
docker build -t chicken-weight-estimator:latest .

# Run the container
docker run -d \
  --name chicken-weight-estimator \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/model_artifacts:/app/model_artifacts:ro \
  -v $(pwd)/config:/app/config:ro \
  chicken-weight-estimator:latest
```

### Option 3: Using Docker Compose (Production)

```bash
# Start all services
docker-compose up -d

# Start with monitoring stack
docker-compose --profile monitoring up -d

# Start with nginx reverse proxy
docker-compose --profile production up -d
```

## 🏗️ Build Options

### Development Build
```powershell
.\build-docker.ps1 -Test
```

### Production Build with Registry Push
```powershell
.\build-docker.ps1 -RegistryUrl "your-registry.com" -ImageTag "v1.0.0"
```

### Build Without Cache
```bash
docker build --no-cache -t chicken-weight-estimator:latest .
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Flask environment |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |
| `LOG_LEVEL` | `INFO` | Logging level |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./logs` | `/app/logs` | Application logs |
| `./model_artifacts` | `/app/model_artifacts` | Model files (read-only) |
| `./config` | `/app/config` | Configuration files (read-only) |

### Port Mapping

| Host Port | Container Port | Service |
|-----------|----------------|---------|
| `8080` | `8080` | Main API |
| `80` | `80` | Nginx (with production profile) |
| `9090` | `9090` | Prometheus (with monitoring profile) |
| `3000` | `3000` | Grafana (with monitoring profile) |

## 📡 API Endpoints

Once running, the following endpoints are available:

- **API Documentation**: http://localhost:8080/
- **Health Check**: http://localhost:8080/ping
- **Detailed Health**: http://localhost:8080/health
- **Demo**: http://localhost:8080/demo
- **Statistics**: http://localhost:8080/stats
- **Main Inference**: http://localhost:8080/invocations (POST)

## 🧪 Testing the Container

### Health Check
```bash
curl http://localhost:8080/ping
```

### Demo Request
```bash
curl http://localhost:8080/demo
```

### Full API Test
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "stream_data": {
      "frame": "<base64-encoded-image>",
      "camera_id": "test_camera",
      "frame_sequence": 1
    }
  }'
```

## 📊 Monitoring

### View Container Logs
```bash
docker logs -f chicken-weight-estimator
```

### Container Statistics
```bash
docker stats chicken-weight-estimator
```

### Health Check Status
```bash
docker inspect --format='{{.State.Health.Status}}' chicken-weight-estimator
```

## 🔧 Troubleshooting

### Container Won't Start

1. **Check Docker is running**:
   ```bash
   docker info
   ```

2. **Check port availability**:
   ```bash
   netstat -an | findstr :8080
   ```

3. **View container logs**:
   ```bash
   docker logs chicken-weight-estimator
   ```

### Performance Issues

1. **Increase memory limit**:
   ```bash
   docker run --memory=4g chicken-weight-estimator:latest
   ```

2. **Check resource usage**:
   ```bash
   docker stats
   ```

### Model Loading Errors

1. **Verify model files exist**:
   ```bash
   ls -la model_artifacts/
   ```

2. **Check file permissions**:
   ```bash
   docker exec chicken-weight-estimator ls -la /app/model_artifacts/
   ```

## 🏭 Production Deployment

### Using Docker Compose with All Services

```bash
# Start full production stack
docker-compose --profile production --profile monitoring up -d

# Check all services
docker-compose ps

# View logs
docker-compose logs -f chicken-weight-estimator
```

### Environment-Specific Configurations

Create environment-specific compose files:

**docker-compose.prod.yml**:
```yaml
version: '3.8'
services:
  chicken-weight-estimator:
    environment:
      - FLASK_ENV=production
      - LOG_LEVEL=WARNING
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

Run with:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Load Balancing

For high availability, use multiple container instances:

```yaml
version: '3.8'
services:
  chicken-weight-estimator:
    deploy:
      replicas: 3
    # ... other configuration
```

## 🔒 Security Considerations

### Container Security
- Runs as non-root user (`appuser`)
- Read-only model artifacts mount
- Limited resource allocation
- Health checks enabled

### Network Security
- Rate limiting via nginx
- Security headers configured
- Internal network isolation

### Secrets Management
```bash
# Use Docker secrets for sensitive data
echo "your-secret" | docker secret create api-key -
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale to 3 instances
docker-compose up -d --scale chicken-weight-estimator=3
```

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## 🔄 Updates and Maintenance

### Update Container
```bash
# Pull latest image
docker pull chicken-weight-estimator:latest

# Recreate container
docker-compose up -d --force-recreate chicken-weight-estimator
```

### Backup Data
```bash
# Backup logs
docker cp chicken-weight-estimator:/app/logs ./backup/logs

# Backup configuration
docker cp chicken-weight-estimator:/app/config ./backup/config
```

### Cleanup
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune
```

## 📞 Support

For issues with Docker deployment:

1. Check the container logs first
2. Verify all prerequisites are met
3. Test with the demo endpoint
4. Check resource availability
5. Review the troubleshooting section

## 🎯 Next Steps

After successful containerization:

1. **Deploy to Cloud**: Use AWS ECS, Azure Container Instances, or Google Cloud Run
2. **Set up CI/CD**: Automate builds and deployments
3. **Add Monitoring**: Integrate with your monitoring stack
4. **Load Testing**: Test with realistic workloads
5. **Security Scanning**: Scan images for vulnerabilities

---

🎉 **Your chicken weight estimation system is now containerized and ready for production deployment!**