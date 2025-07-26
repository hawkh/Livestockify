# PowerShell script to run the chicken weight estimation Docker container
param(
    [string]$ImageName = "chicken-weight-estimator:latest",
    [int]$Port = 8080,
    [string]$ContainerName = "chicken-weight-estimator",
    [switch]$Detached = $false,
    [switch]$Remove = $false,
    [switch]$Build = $false
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

function Write-Info {
    param([string]$Message)
    Write-Host "${Blue}ℹ️  $Message${Reset}"
}

function Write-Error {
    param([string]$Message)
    Write-Host "${Red}❌ $Message${Reset}"
}

Write-Host "${Blue}🐔 Running Chicken Weight Estimation Docker Container${Reset}"
Write-Host "=================================================="

# Build image if requested
if ($Build) {
    Write-Info "Building Docker image first..."
    $imageParts = $ImageName -split ':'
    $imageNameOnly = $imageParts[0]
    $imageTag = if ($imageParts.Length -gt 1) { $imageParts[1] } else { 'latest' }
    
    & .\build-docker-fixed.ps1 -ImageName $imageNameOnly -ImageTag $imageTag
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed"
        exit 1
    }
}

# Stop and remove existing container if it exists
if ($Remove) {
    Write-Info "Stopping and removing existing container..."
    docker stop $ContainerName 2>$null | Out-Null
    docker rm $ContainerName 2>$null | Out-Null
}

# Prepare docker run arguments
$runArgs = @(
    "run"
    "--name", $ContainerName
    "-p", "${Port}:8080"
    "-v", "${PWD}/logs:/app/logs"
    "-v", "${PWD}/model_artifacts:/app/model_artifacts:ro"
    "-v", "${PWD}/config:/app/config:ro"
)

if ($Detached) {
    $runArgs += "-d"
} else {
    $runArgs += "--rm", "-it"
}

$runArgs += $ImageName

# Run the container
Write-Info "Starting container with command:"
Write-Host "docker $($runArgs -join ' ')"
Write-Host ""

& docker @runArgs

if ($LASTEXITCODE -eq 0) {
    Write-Status "Container started successfully!"
    
    if ($Detached) {
        Write-Host ""
        Write-Host "${Blue}🌐 Access Points${Reset}"
        Write-Host "=================================================="
        Write-Host "API Documentation: http://localhost:${Port}/"
        Write-Host "Health Check:      http://localhost:${Port}/ping"
        Write-Host "Demo Endpoint:     http://localhost:${Port}/demo"
        Write-Host "Statistics:        http://localhost:${Port}/stats"
        Write-Host ""
        Write-Host "${Blue}📊 Management Commands${Reset}"
        Write-Host "=================================================="
        Write-Host "View logs:    docker logs -f $ContainerName"
        Write-Host "Stop container: docker stop $ContainerName"
        Write-Host "Remove container: docker rm $ContainerName"
        Write-Host ""
        
        # Wait a moment and test the health endpoint
        Write-Info "Testing container health..."
        Start-Sleep -Seconds 10
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:${Port}/ping" -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Status "Container is healthy and responding!"
            }
        } catch {
            Write-Host "${Yellow}⚠️  Container may still be starting up. Check logs if issues persist.${Reset}"
        }
    }
} else {
    Write-Error "Failed to start container"
    exit 1
}