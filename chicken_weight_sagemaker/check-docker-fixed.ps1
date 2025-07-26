# Check Docker Desktop status and provide installation guidance
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Blue = "`e[34m"
$Reset = "`e[0m"

Write-Host "${Blue}🐔 Docker Desktop Status Check${Reset}"
Write-Host "=================================================="

# Check if Docker command is available
Write-Host "Checking Docker installation..."
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "${Green}✅ Docker CLI is installed: $dockerVersion${Reset}"
    } else {
        throw "Docker CLI not found"
    }
} catch {
    Write-Host "${Red}❌ Docker CLI is not installed${Reset}"
    Write-Host ""
    Write-Host "${Blue}📥 Installation Instructions:${Reset}"
    Write-Host "1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop"
    Write-Host "2. Run the installer and follow the setup wizard"
    Write-Host "3. Restart your computer if prompted"
    Write-Host "4. Start Docker Desktop from the Start menu"
    Write-Host "5. Wait for Docker to start (you'll see the whale icon in system tray)"
    Write-Host "6. Run this script again to verify installation"
    exit 1
}

# Check if Docker daemon is running
Write-Host "Checking Docker daemon status..."
try {
    docker info > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "${Green}✅ Docker daemon is running${Reset}"
        
        # Get Docker system info
        Write-Host ""
        Write-Host "${Blue}📊 Docker System Information:${Reset}"
        docker system df
        
    } else {
        throw "Docker daemon not running"
    }
} catch {
    Write-Host "${Red}❌ Docker daemon is not running${Reset}"
    Write-Host ""
    Write-Host "${Blue}🚀 How to Start Docker Desktop:${Reset}"
    Write-Host "1. Look for Docker Desktop in your Start menu"
    Write-Host "2. Click on Docker Desktop to start it"
    Write-Host "3. Wait for the whale icon to appear in your system tray"
    Write-Host "4. The icon should be steady (not animated) when ready"
    Write-Host "5. You can also right-click the tray icon to see status"
    Write-Host ""
    Write-Host "${Yellow}⚠️  If Docker Desktop is not installed:${Reset}"
    Write-Host "   Download from: https://www.docker.com/products/docker-desktop"
    Write-Host ""
    Write-Host "${Yellow}⚠️  If you're having issues:${Reset}"
    Write-Host "   1. Try restarting Docker Desktop"
    Write-Host "   2. Check Windows features: Hyper-V and Containers"
    Write-Host "   3. Ensure virtualization is enabled in BIOS"
    Write-Host "   4. Try running as Administrator"
    exit 1
}

Write-Host ""
Write-Host "${Green}🎉 Docker is ready for use!${Reset}"
Write-Host ""
Write-Host "${Blue}📝 Next Steps:${Reset}"
Write-Host "1. Build the container: .\build-docker-fixed.ps1"
Write-Host "2. Run the container: .\run-docker-fixed.ps1 -Detached"
Write-Host "3. Test the API: curl http://localhost:8080/ping"
Write-Host ""
Write-Host "${Blue}💡 Useful Docker Commands:${Reset}"
Write-Host "- List running containers: docker ps"
Write-Host "- List all containers: docker ps -a"
Write-Host "- List images: docker images"
Write-Host "- Remove unused containers: docker container prune"
Write-Host "- Remove unused images: docker image prune"