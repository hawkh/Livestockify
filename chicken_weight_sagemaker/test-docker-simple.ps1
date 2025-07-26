# Simple Docker setup test
Write-Host "🐔 Testing Docker Setup for Chicken Weight Estimation" -ForegroundColor Blue
Write-Host "=================================================="

# Test 1: Check if Dockerfile exists
if (Test-Path "Dockerfile") {
    Write-Host "✅ Dockerfile exists" -ForegroundColor Green
} else {
    Write-Host "❌ Dockerfile not found" -ForegroundColor Red
}

# Test 2: Check docker-compose.yml
if (Test-Path "docker-compose.yml") {
    Write-Host "✅ docker-compose.yml exists" -ForegroundColor Green
} else {
    Write-Host "⚠️  docker-compose.yml not found" -ForegroundColor Yellow
}

# Test 3: Check required directories
$requiredDirs = @("src", "logs", "model_artifacts", "config")
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "✅ Directory exists: $dir" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Directory missing: $dir" -ForegroundColor Yellow
    }
}

# Test 4: Check key application files
$appFiles = @(
    "demo_server.py",
    "src/inference/stream_handler.py",
    "docker/requirements.txt"
)

foreach ($file in $appFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Application file exists: $file" -ForegroundColor Green
    } else {
        Write-Host "❌ Critical file missing: $file" -ForegroundColor Red
    }
}

# Test 5: Check Docker CLI
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker CLI available: $dockerVersion" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Docker CLI not available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Docker CLI not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📋 Summary" -ForegroundColor Blue
Write-Host "=================================================="

if ((Test-Path "Dockerfile") -and (Test-Path "demo_server.py") -and (Test-Path "docker/requirements.txt")) {
    Write-Host "✅ Docker setup is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Next Steps:" -ForegroundColor Blue
    Write-Host "1. Start Docker Desktop"
    Write-Host "2. Run: .\build-docker.ps1"
    Write-Host "3. Run: .\run-docker.ps1 -Detached"
    Write-Host "4. Test: curl http://localhost:8080/ping"
} else {
    Write-Host "❌ Docker setup has missing components" -ForegroundColor Red
}

Write-Host ""
Write-Host "📚 See DOCKER_README.md for detailed instructions" -ForegroundColor Blue