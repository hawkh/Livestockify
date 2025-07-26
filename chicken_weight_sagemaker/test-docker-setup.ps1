# Test script to verify Docker setup without building
param(
    [switch]$Verbose = $false
)

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

Write-Host "${Blue}🐔 Testing Docker Setup for Chicken Weight Estimation${Reset}"
Write-Host "=================================================="

# Test 1: Check if Dockerfile exists and is valid
Write-Info "Testing Dockerfile..."
if (Test-Path "Dockerfile") {
    Write-Status "Dockerfile exists"
    
    # Check Dockerfile content
    $dockerfileContent = Get-Content "Dockerfile" -Raw
    
    $checks = @(
        @{ Pattern = "FROM python:3.11-slim"; Description = "Base image specified" },
        @{ Pattern = "WORKDIR /app"; Description = "Working directory set" },
        @{ Pattern = "COPY.*requirements.txt"; Description = "Requirements file copied" },
        @{ Pattern = "RUN pip install"; Description = "Dependencies installation" },
        @{ Pattern = "EXPOSE 8080"; Description = "Port exposed" },
        @{ Pattern = "HEALTHCHECK"; Description = "Health check configured" }
    )
    
    foreach ($check in $checks) {
        if ($dockerfileContent -match $check.Pattern) {
            Write-Status $check.Description
        } else {
            Write-Warning "$($check.Description) - Not found or different pattern"
        }
    }
} else {
    Write-Error "Dockerfile not found"
}

# Test 2: Check docker-compose.yml
Write-Info "Testing Docker Compose configuration..."
if (Test-Path "docker-compose.yml") {
    Write-Status "docker-compose.yml exists"
    
    $composeContent = Get-Content "docker-compose.yml" -Raw
    
    $composeChecks = @(
        @{ Pattern = "version:.*3\.8"; Description = "Compose version specified" },
        @{ Pattern = "chicken-weight-estimator:"; Description = "Main service defined" },
        @{ Pattern = "ports:.*8080:8080"; Description = "Port mapping configured" },
        @{ Pattern = "volumes:"; Description = "Volume mounts configured" },
        @{ Pattern = "healthcheck:"; Description = "Health check in compose" }
    )
    
    foreach ($check in $composeChecks) {
        if ($composeContent -match $check.Pattern) {
            Write-Status $check.Description
        } else {
            Write-Warning "$($check.Description) - Not found"
        }
    }
} else {
    Write-Warning "docker-compose.yml not found"
}

# Test 3: Check required directories
Write-Info "Testing directory structure..."
$requiredDirs = @("src", "logs", "model_artifacts", "config", "nginx", "monitoring")

foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Status "Directory exists: $dir"
    } else {
        Write-Warning "Directory missing: $dir"
    }
}

# Test 4: Check configuration files
Write-Info "Testing configuration files..."
$configFiles = @(
    "config/model_config.yaml",
    "config/camera_config.yaml",
    "nginx/nginx.conf",
    "monitoring/prometheus.yml"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        Write-Status "Config file exists: $file"
    } else {
        Write-Warning "Config file missing: $file"
    }
}

# Test 5: Check .dockerignore
Write-Info "Testing .dockerignore..."
if (Test-Path ".dockerignore") {
    Write-Status ".dockerignore exists"
    
    $dockerignoreContent = Get-Content ".dockerignore" -Raw
    
    $ignoreChecks = @(
        @{ Pattern = "__pycache__"; Description = "Python cache ignored" },
        @{ Pattern = "\.git"; Description = "Git files ignored" },
        @{ Pattern = "\.venv"; Description = "Virtual environment ignored" },
        @{ Pattern = "tests/"; Description = "Test files ignored" }
    )
    
    foreach ($check in $ignoreChecks) {
        if ($dockerignoreContent -match $check.Pattern) {
            Write-Status $check.Description
        } else {
            Write-Warning "$($check.Description) - Not found in .dockerignore"
        }
    }
} else {
    Write-Warning ".dockerignore not found"
}

# Test 6: Check build scripts
Write-Info "Testing build scripts..."
$buildScripts = @("build-docker.ps1", "run-docker.ps1", "build-docker.sh")

foreach ($script in $buildScripts) {
    if (Test-Path $script) {
        Write-Status "Build script exists: $script"
    } else {
        Write-Warning "Build script missing: $script"
    }
}

# Test 7: Check application files
Write-Info "Testing application files..."
$appFiles = @(
    "demo_server.py",
    "src/inference/stream_handler.py",
    "src/utils/config/config_manager.py",
    "docker/requirements.txt"
)

foreach ($file in $appFiles) {
    if (Test-Path $file) {
        Write-Status "Application file exists: $file"
    } else {
        Write-Error "Critical application file missing: $file"
    }
}

# Test 8: Validate requirements.txt
Write-Info "Testing requirements.txt..."
if (Test-Path "docker/requirements.txt") {
    $requirements = Get-Content "docker/requirements.txt"
    
    $requiredPackages = @("torch", "opencv-python", "flask", "numpy", "ultralytics")
    
    foreach ($package in $requiredPackages) {
        $found = $requirements | Where-Object { $_ -match "^$package" }
        if ($found) {
            Write-Status "Required package found: $package"
        } else {
            Write-Warning "Required package missing: $package"
        }
    }
} else {
    Write-Error "requirements.txt not found"
}

# Test 9: Check Docker availability (without requiring it to be running)
Write-Info "Testing Docker availability..."
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Docker CLI available: $dockerVersion"
    } else {
        Write-Warning "Docker CLI not available"
    }
} catch {
    Write-Warning "Docker CLI not found in PATH"
}

# Summary
Write-Host ""
Write-Host "${Blue}📋 Docker Setup Summary${Reset}"
Write-Host "=================================================="

$readyToBuild = $true

# Check critical files
$criticalFiles = @("Dockerfile", "demo_server.py", "docker/requirements.txt")
foreach ($file in $criticalFiles) {
    if (!(Test-Path $file)) {
        Write-Error "Critical file missing: $file"
        $readyToBuild = $false
    }
}

if ($readyToBuild) {
    Write-Status "Docker setup is ready!"
    Write-Host ""
    Write-Host "${Blue}🚀 Next Steps${Reset}"
    Write-Host "=================================================="
    Write-Host "1. Start Docker Desktop"
    Write-Host "2. Run: .\build-docker.ps1"
    Write-Host "3. Run: .\run-docker.ps1 -Detached"
    Write-Host "4. Test: curl http://localhost:8080/ping"
    Write-Host ""
    Write-Host "Or use Docker Compose:"
    Write-Host "  docker-compose up -d"
} else {
    Write-Error "Docker setup has issues that need to be resolved"
}

Write-Host ""
Write-Host "${Blue}📚 Documentation${Reset}"
Write-Host "=================================================="
Write-Host "See DOCKER_README.md for detailed instructions"