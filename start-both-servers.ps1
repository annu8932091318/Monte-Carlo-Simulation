# Start both Python and Node.js servers with monitoring
Write-Host "🚀 Starting Risk Simulation Dual-Engine Servers" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Kill any existing processes
Write-Host "🧹 Cleaning up existing processes..." -ForegroundColor Yellow

# Kill Python server on port 3002
$pythonProcess = Get-NetTCPConnection -LocalPort 3002 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($pythonProcess) {
    Stop-Process -Id $pythonProcess -Force -ErrorAction SilentlyContinue
    Write-Host "  ✅ Stopped existing Python server" -ForegroundColor Green
}

# Kill Node.js server on port 5010
$nodeProcess = Get-NetTCPConnection -LocalPort 5010 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($nodeProcess) {
    Stop-Process -Id $nodeProcess -Force -ErrorAction SilentlyContinue
    Write-Host "  ✅ Stopped existing Node.js server" -ForegroundColor Green
}

Write-Host ""

# Start Python server
Write-Host "🐍 Starting Python Flask Server..." -ForegroundColor Blue
$pythonPath = "C:\Users\annuy\OneDrive\Documents\projects\Monte-Carlo-Simulation\backend\python"
Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $pythonPath -WindowStyle Minimized

# Wait a moment for Python server to start
Start-Sleep -Seconds 3

# Start Node.js server
Write-Host "🟢 Starting Node.js Express Server..." -ForegroundColor Blue
$nodePath = "C:\Users\annuy\OneDrive\Documents\projects\Monte-Carlo-Simulation\backend\node"
Start-Process -FilePath "npm" -ArgumentList "start" -WorkingDirectory $nodePath -WindowStyle Minimized

# Wait a moment for Node.js server to start
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "🎯 Servers Started! Checking status..." -ForegroundColor Green

# Check if servers are running
for ($i = 1; $i -le 10; $i++) {
    Write-Host "⏱️  Status check $i/10..." -ForegroundColor Yellow
    
    $pythonRunning = Test-NetConnection -ComputerName localhost -Port 3002 -InformationLevel Quiet -WarningAction SilentlyContinue
    $nodeRunning = Test-NetConnection -ComputerName localhost -Port 5010 -InformationLevel Quiet -WarningAction SilentlyContinue
    
    $pythonStatus = if ($pythonRunning) { "✅ RUNNING" } else { "❌ NOT RUNNING" }
    $nodeStatus = if ($nodeRunning) { "✅ RUNNING" } else { "❌ NOT RUNNING" }
    
    Write-Host "  🐍 Python Server (port 3002): $pythonStatus" -ForegroundColor $(if ($pythonRunning) { "Green" } else { "Red" })
    Write-Host "  🟢 Node.js Server (port 5010): $nodeStatus" -ForegroundColor $(if ($nodeRunning) { "Green" } else { "Red" })
    
    if ($pythonRunning -and $nodeRunning) {
        Write-Host ""
        Write-Host "🎉 Both servers are running successfully!" -ForegroundColor Green
        Write-Host "🔗 Python API: http://localhost:3002/health" -ForegroundColor Cyan
        Write-Host "🔗 Node.js API: http://localhost:5010/health" -ForegroundColor Cyan
        Write-Host "📖 Python Docs: http://localhost:3002/api/docs" -ForegroundColor Cyan
        Write-Host "📖 Node.js Docs: http://localhost:5010/api/docs" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "✅ Dual-engine setup complete! Both servers are stable." -ForegroundColor Green
        break
    }
    
    if ($i -eq 10) {
        Write-Host ""
        Write-Host "⚠️  Some servers may not be fully started yet. Check manually." -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "💡 Servers are running in the background. Use Task Manager to stop them if needed." -ForegroundColor Gray
Write-Host "💡 Or run 'Get-Process python,node | Stop-Process' to stop all servers." -ForegroundColor Gray
