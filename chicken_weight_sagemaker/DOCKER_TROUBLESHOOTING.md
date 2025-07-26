# 🐔 Docker Troubleshooting Guide

This guide helps resolve common Docker issues when building and running the chicken weight estimation system.

## 🚨 Common Issues and Solutions

### Issue 1: "Docker is not available" or "command not found"

**Error Message:**
```
ERROR: error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

**Solution:**
1. **Install Docker Desktop:**
   - Download from: https://www.docker.com/products/docker-desktop
   - Run the installer as Administrator
   - Restart your computer when prompted

2. **Start Docker Desktop:**
   - Look for Docker Desktop in Start menu
   - Click to start it
   - Wait for the whale icon in system tray
   - Icon should be steady (not animated) when ready

3. **Verify Installation:**
   ```powershell
   .\check-docker.ps1
   ```

### Issue 2: PowerShell Script Syntax Errors

**Error Message:**
```
Missing closing '}' in statement block or type definition
The string is missing the terminator: "
```

**Solution:**
Use the fixed scripts:
```powershell
# Instead of .\build-docker.ps1
.\build-docker-fixed.ps1

# Instead of .\run-docker.ps1
.\run-docker-fixed.ps1 -Detached
```

Or use the batch file alternative:
```cmd
build-docker.bat
```

### Issue 3: Docker Daemon Not Running

**Error Message:**
```
docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution:**
1. **Check Docker Desktop Status:**
   - Look for whale icon in system tray
   - Right-click icon → check status
   - Should show "Docker Desktop is running"

2. **Restart Docker Desktop:**
   - Right-click whale icon → "Restart Docker Desktop"
   - Wait for restart to complete

3. **Check Windows Services:**
   - Press Win+R → type `services.msc`
   - Look for "Docker Desktop Service"
   - Ensure it's running

### Issue 4: Permission Denied Errors

**Error Message:**
```
permission denied while trying to connect to the Docker daemon socket
```

**Solution:**
1. **Run as Administrator:**
   - Right-click PowerShell → "Run as Administrator"
   - Try the build command again

2. **Check Docker Desktop Settings:**
   - Open Docker Desktop
   - Go to Settings → General
   - Ensure "Use the WSL 2 based engine" is checked (if available)

### Issue 5: Build Context Too Large

**Error Message:**
```
Sending build context to Docker daemon  XXX.XGB
```

**Solution:**
1. **Check .dockerignore:**
   ```powershell
   Get-Content .dockerignore
   ```

2. **Clean up large files:**
   ```powershell
   # Remove large test files
   Remove-Item -Recurse -Force test_output -ErrorAction SilentlyContinue
   Remove-Item -Recurse -Force *.log -ErrorAction SilentlyContinue
   ```

### Issue 6: Port Already in Use

**Error Message:**
```
bind: address already in use
```

**Solution:**
1. **Find what's using the port:**
   ```powershell
   netstat -ano | findstr :8080
   ```

2. **Stop existing container:**
   ```powershell
   docker stop chicken-weight-estimator
   docker rm chicken-weight-estimator
   ```

3. **Use different port:**
   ```powershell
   .\run-docker-fixed.ps1 -Port 8081 -Detached
   ```

### Issue 7: Out of Disk Space

**Error Message:**
```
no space left on device
```

**Solution:**
1. **Clean Docker system:**
   ```powershell
   docker system prune -a
   ```

2. **Remove unused images:**
   ```powershell
   docker image prune -a
   ```

3. **Check disk space:**
   ```powershell
   docker system df
   ```

## 🔧 Diagnostic Commands

### Check Docker Status
```powershell
# Run comprehensive check
.\check-docker.ps1

# Basic checks
docker --version
docker info
docker ps
```

### View Container Logs
```powershell
# View logs
docker logs chicken-weight-estimator

# Follow logs in real-time
docker logs -f chicken-weight-estimator

# View last 50 lines
docker logs --tail 50 chicken-weight-estimator
```

### Debug Container Issues
```powershell
# Run container interactively
docker run -it --rm chicken-weight-estimator:latest /bin/bash

# Execute command in running container
docker exec -it chicken-weight-estimator /bin/bash

# Inspect container
docker inspect chicken-weight-estimator
```

### Check Resource Usage
```powershell
# View container stats
docker stats chicken-weight-estimator

# View system resource usage
docker system df
docker system events
```

## 🚀 Alternative Installation Methods

### Method 1: Using Windows Package Manager
```powershell
# Install using winget (Windows 10/11)
winget install Docker.DockerDesktop
```

### Method 2: Using Chocolatey
```powershell
# Install Chocolatey first, then:
choco install docker-desktop
```

### Method 3: Manual Installation
1. Download Docker Desktop installer
2. Run as Administrator
3. Follow installation wizard
4. Restart computer
5. Start Docker Desktop

## 🔍 System Requirements

### Minimum Requirements:
- **OS:** Windows 10 64-bit (Pro, Enterprise, or Education)
- **RAM:** 4GB minimum (8GB recommended)
- **CPU:** 64-bit processor with Second Level Address Translation (SLAT)
- **Virtualization:** Enabled in BIOS/UEFI
- **Hyper-V:** Enabled (Windows feature)

### Check System Compatibility:
```powershell
# Check Windows version
Get-ComputerInfo | Select WindowsProductName, WindowsVersion

# Check if Hyper-V is enabled
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All

# Check virtualization support
systeminfo | findstr /i "hyper-v"
```

## 📞 Getting Help

### If you're still having issues:

1. **Check Docker Desktop Logs:**
   - Open Docker Desktop
   - Go to Settings → Troubleshoot
   - Click "Get support" for logs

2. **Community Resources:**
   - Docker Community Forums: https://forums.docker.com/
   - Stack Overflow: https://stackoverflow.com/questions/tagged/docker
   - Docker Documentation: https://docs.docker.com/

3. **Project-Specific Help:**
   - Check the main README.md
   - Review DOCKER_README.md for detailed instructions
   - Run the test scripts to verify functionality

## 🎯 Quick Fix Checklist

Before asking for help, try these steps:

- [ ] Docker Desktop is installed and running
- [ ] Whale icon is visible and steady in system tray
- [ ] `docker --version` works in terminal
- [ ] `docker info` shows system information
- [ ] No other containers using port 8080
- [ ] Sufficient disk space available
- [ ] Running terminal as Administrator (if needed)
- [ ] Using the fixed PowerShell scripts
- [ ] Firewall/antivirus not blocking Docker

## 🔄 Complete Reset Procedure

If nothing else works, try a complete reset:

1. **Uninstall Docker Desktop:**
   ```powershell
   # Stop Docker Desktop
   # Uninstall from Control Panel
   ```

2. **Clean up remaining files:**
   ```powershell
   Remove-Item -Recurse -Force "$env:APPDATA\Docker" -ErrorAction SilentlyContinue
   Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Docker" -ErrorAction SilentlyContinue
   ```

3. **Reinstall Docker Desktop:**
   - Download fresh installer
   - Run as Administrator
   - Restart computer
   - Start Docker Desktop

4. **Test installation:**
   ```powershell
   .\check-docker.ps1
   ```

---

🎉 **Once Docker is working, you can build and run your chicken weight estimation system successfully!**