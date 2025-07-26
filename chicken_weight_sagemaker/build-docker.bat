@echo off
echo 🐔 Building Chicken Weight Estimation Docker Container
echo ==================================================

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not available or not running
    echo Please ensure Docker Desktop is installed and running
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo ✅ Docker is available

REM Check if Docker daemon is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker daemon is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo ✅ Docker daemon is running

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "model_artifacts" mkdir model_artifacts
if not exist "config" mkdir config
if not exist "nginx" mkdir nginx
if not exist "monitoring" mkdir monitoring

REM Copy config files if they don't exist
if not exist "config\model_config.yaml" (
    if exist "src\utils\config\model_config.yaml" (
        copy "src\utils\config\model_config.yaml" "config\"
        echo ✅ Copied model_config.yaml
    )
)

if not exist "config\camera_config.yaml" (
    if exist "src\utils\config\camera_config.yaml" (
        copy "src\utils\config\camera_config.yaml" "config\"
        echo ✅ Copied camera_config.yaml
    )
)

REM Build the Docker image
echo 🔨 Building Docker image...
echo Image: chicken-weight-estimator:latest

docker build --tag chicken-weight-estimator:latest --progress=plain .

if %errorlevel% eq 0 (
    echo ✅ Docker image built successfully
) else (
    echo ❌ Docker build failed
    pause
    exit /b 1
)

REM Show image information
echo 📋 Build Summary
echo ==================================================
docker images chicken-weight-estimator:latest

echo.
echo 🚀 Usage Instructions
echo ==================================================
echo Run the container:
echo   docker run -p 8080:8080 chicken-weight-estimator:latest
echo.
echo Run with Docker Compose:
echo   docker-compose up -d
echo.
echo View logs:
echo   docker logs -f chicken-weight-estimator
echo.
echo Access the API:
echo   http://localhost:8080/
echo.

echo ✅ Build process completed successfully!
echo 🎉 Your chicken weight estimation system is ready to deploy!

pause