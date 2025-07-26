# PowerShell build script for chicken weight estimation Docker container
param(
    [string]$ImageName = "chicken-weight-estimator",
    [string]$ImageTag = "latest",
    [string]$RegistryUrl = "",
    [switch]$NoBuild = $false,
    [switch]$Test = $false
)

# Colors for output
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-Status {
    param([string]$Message)
    Write-Host "${Green}✅ $Message${Reset}"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "${Yellow}⚠️  $Message${Reset}"
}

function Write-Error {
    param([string]$Message)
    Write-Host "${Red}❌ $Message${Reset}"
}

function Write-Info {
    param([string]$Message)
    Write-Host "${Blue}ℹ️  $Message${Reset}"
}

Write-Host "${Blue}🐔 Building Chicken Weight Estimation Docker Container${Reset}"
Write-Host "=================================================="

# Check if Docker is available
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed"
    }
    Write-Status "Docker is available: $dockerVersion"
} catch {
    Write-Error "Docker is not available or not running."
    Write-Info "Please ensure Docker Desktop is installed and running."
    Write-Info "Download from: https://www.docker.com/products/docker-desktop"
    exit 1
}

# Check if Docker daemon is running
try {
    docker info > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker daemon not running"
    }
    Write-Status "Docker daemon is running"
} catch {
    Write-Error "Docker daemon is not running."
    Write-Info "Please start Docker Desktop and try again."
    exit 1
}

# Create necessary directories
Write-Info "📁 Creating necessary directories..."
$directories = @("logs", "model_artifacts", "config", "nginx", "monitoring")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Status "Created directory: $dir"
    }
}

# Copy config files if they don't exist
if (!(Test-Path "config/model_config.yaml")) {
    if (Test-Path "src/utils/config/model_config.yaml") {
        Copy-Item "src/utils/config/model_config.yaml" "config/"
        Write-Status "Copied model_config.yaml"
    }
}

if (!(Test-Path "config/camera_config.yaml")) {
    if (Test-Path "src/utils/config/camera_config.yaml") {
        Copy-Item "src/utils/config/camera_config.yaml" "config/"
        Write-Status "Copied camera_config.yaml"
    }
}

if (!$NoBuild) {
    # Build the Docker image
    Write-Info "🔨 Building Docker image..."
    Write-Host "Image: ${ImageName}:${ImageTag}"

    $buildArgs = @(
        "build",
        "--tag", "${ImageName}:${ImageTag}",
        "--progress=plain",
        "."
    )

    Write-Info "Running: docker $($buildArgs -join ' ')"
    
    & docker @buildArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Docker image built successfully"
    } else {
        Write-Error "Docker build failed with exit code $LASTEXITCODE"
        exit 1
    }

    # Get image information
    $imageInfo = docker images "${ImageName}:${ImageTag}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | Select-Object -Skip 1
    Write-Status "Image info: $imageInfo"
}

if ($Test) {
    # Test the container
    Write-Info "🧪 Testing the container..."

    # Stop any existing test container
    docker stop chicken-test 2>$null | Out-Null
    docker rm chicken-test 2>$null | Out-Null

    # Run test container
    Write-Info "Starting test container..."
    docker run --rm --name chicken-test -p 8081:8080 -d "${ImageName}:${ImageTag}"

    if ($LASTEXITCODE -eq 0) {
        Write-Status "Test container started"
        
        # Wait for container to be ready
        Write-Info "Waiting for container to start (30 seconds)..."
        Start-Sleep -Seconds 30

        # Test health endpoint
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8081/ping" -TimeoutSec 10 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Status "Container health check passed"
            } else {
                Write-Warning "Container health check returned status: $($response.StatusCode)"
            }
        } catch {
            Write-Warning "Container health check failed: $($_.Exception.Message)"
            Write-Info "This might be expected if the container is still starting up"
        }

        # Show container logs
        Write-Info "Container logs:"
        docker logs chicken-test

        # Stop test container
        Write-Info "Stopping test container..."
        docker stop chicken-test | Out-Null
        Write-Status "Test container stopped"
    } else {
        Write-Error "Failed to start test container"
    }
}

# Show final information
Write-Host "${Blue}📋 Build Summary${Reset}"
Write-Host "=================================================="
Write-Host "Image Name: ${ImageName}:${ImageTag}"
Write-Host "Build Date: $(Get-Date)"

# Show usage instructions
Write-Host "${Blue}🚀 Usage Instructions${Reset}"
Write-Host "=================================================="
Write-Host "Run the container:"
Write-Host "  docker run -p 8080:8080 ${ImageName}:${ImageTag}"
Write-Host ""
Write-Host "Run with Docker Compose:"
Write-Host "  docker-compose up -d"
Write-Host ""
Write-Host "View logs:"
Write-Host "  docker logs -f chicken-weight-estimator"
Write-Host ""
Write-Host "Access the API:"
Write-Host "  http://localhost:8080/"

# Optional: Push to registry
if ($RegistryUrl) {
    Write-Info "📤 Pushing to registry..."
    
    # Tag for registry
    docker tag "${ImageName}:${ImageTag}" "${RegistryUrl}/${ImageName}:${ImageTag}"
    
    # Push to registry
    docker push "${RegistryUrl}/${ImageName}:${ImageTag}"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Image pushed to registry: ${RegistryUrl}/${ImageName}:${ImageTag}"
    } else {
        Write-Error "Failed to push image to registry"
    }
}

Write-Status "Build process completed successfully!"
Write-Host "${Green}🎉 Your chicken weight estimation system is ready to deploy!${Reset}"

# Show next steps
Write-Host "${Blue}📝 Next Steps${Reset}"
Write-Host "=================================================="
Write-Host "1. Start the container: docker run -p 8080:8080 ${ImageName}:${ImageTag}"
Write-Host "2. Test the API: curl http://localhost:8080/ping"
Write-Host "3. View documentation: http://localhost:8080/"
Write-Host "4. Run demo: http://localhost:8080/demo"
Write-Host "5. For production: docker-compose up -d"