# Risk Simulation Engine - Dual Server Monitor
# Monitors both Python and Node.js backend servers

function Show-ServerStatus {
    Write-Host ""
    Write-Host "🚀 ========================================" -ForegroundColor Green
    Write-Host "🚀    RISK SIMULATION ENGINE STATUS      " -ForegroundColor Green  
    Write-Host "🚀 ========================================" -ForegroundColor Green
    Write-Host ""

    # Check server connectivity
    $pythonStatus = Test-NetConnection -ComputerName localhost -Port 3002 -InformationLevel Quiet -WarningAction SilentlyContinue
    $nodeStatus = Test-NetConnection -ComputerName localhost -Port 5010 -InformationLevel Quiet -WarningAction SilentlyContinue
    $frontendStatus = Test-NetConnection -ComputerName localhost -Port 3000 -InformationLevel Quiet -WarningAction SilentlyContinue

    Write-Host "📊 SERVER STATUS:" -ForegroundColor Yellow
    Write-Host "   🐍 Python Backend (port 3002):  $(if($pythonStatus){'✅ RUNNING'}else{'❌ STOPPED'})" -ForegroundColor $(if($pythonStatus){'Green'}else{'Red'})
    Write-Host "   🟢 Node.js Backend (port 5010): $(if($nodeStatus){'✅ RUNNING'}else{'❌ STOPPED'})" -ForegroundColor $(if($nodeStatus){'Green'}else{'Red'})
    Write-Host "   🌐 React Frontend (port 3000):  $(if($frontendStatus){'✅ RUNNING'}else{'❌ STOPPED'})" -ForegroundColor $(if($frontendStatus){'Green'}else{'Red'})
    Write-Host ""

    # Test health endpoints if servers are running
    if ($pythonStatus) {
        try {
            $pythonHealth = Invoke-RestMethod -Uri "http://localhost:3002/health" -Method GET -TimeoutSec 5
            Write-Host "🐍 PYTHON SERVER DETAILS:" -ForegroundColor Blue
            Write-Host "   Status: $($pythonHealth.status)" -ForegroundColor Cyan
            Write-Host "   Uptime: $([math]::Round($pythonHealth.uptime, 1)) seconds" -ForegroundColor Cyan
            Write-Host "   URL: http://localhost:3002" -ForegroundColor Cyan
        } catch {
            Write-Host "🐍 Python server connectivity issue: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    if ($nodeStatus) {
        try {
            $nodeHealth = Invoke-RestMethod -Uri "http://localhost:5010/health" -Method GET -TimeoutSec 5
            Write-Host "🟢 NODE.JS SERVER DETAILS:" -ForegroundColor Blue
            Write-Host "   Status: $($nodeHealth.status)" -ForegroundColor Cyan
            Write-Host "   Uptime: $([math]::Round($nodeHealth.uptime, 1)) seconds" -ForegroundColor Cyan
            Write-Host "   Memory: $([math]::Round($nodeHealth.memory_usage.heapUsed / 1024 / 1024, 1)) MB" -ForegroundColor Cyan
            Write-Host "   URL: http://localhost:5010" -ForegroundColor Cyan
        } catch {
            Write-Host "🟢 Node.js server connectivity issue: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Host "🔗 QUICK LINKS:" -ForegroundColor Yellow
    Write-Host "   🌐 Frontend:        http://localhost:3000" -ForegroundColor White
    Write-Host "   🐍 Python API:      http://localhost:3002/api/docs" -ForegroundColor White
    Write-Host "   🟢 Node.js API:     http://localhost:5010/api/docs" -ForegroundColor White
    Write-Host "   🏥 Python Health:   http://localhost:3002/health" -ForegroundColor White
    Write-Host "   🏥 Node.js Health:  http://localhost:5010/health" -ForegroundColor White
    Write-Host ""

    # Show dual-engine status
    if ($pythonStatus -and $nodeStatus) {
        Write-Host "🎉 DUAL-ENGINE MODE: ✅ ACTIVE" -ForegroundColor Green
        Write-Host "   Both backend engines are running successfully!" -ForegroundColor Green
        Write-Host "   You can now use comparison mode in the frontend." -ForegroundColor Green
    } elseif ($pythonStatus -or $nodeStatus) {
        Write-Host "⚠️  SINGLE-ENGINE MODE: PARTIAL" -ForegroundColor Yellow
        Write-Host "   Only one backend engine is running." -ForegroundColor Yellow
    } else {
        Write-Host "❌ NO ENGINES RUNNING" -ForegroundColor Red
        Write-Host "   Neither backend server is responding." -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "🚀 ========================================" -ForegroundColor Green
    Write-Host ""
}

# Show initial status
Show-ServerStatus

# Offer to run a continuous monitor
$monitor = Read-Host "Would you like to start continuous monitoring? (y/n)"
if ($monitor -eq 'y' -or $monitor -eq 'Y') {
    Write-Host "Starting continuous monitoring... Press Ctrl+C to stop." -ForegroundColor Yellow
    Write-Host ""
    
    while ($true) {
        Clear-Host
        Show-ServerStatus
        Write-Host "⏰ Last checked: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
        Write-Host "🔄 Refreshing in 30 seconds... (Press Ctrl+C to stop)" -ForegroundColor Gray
        Start-Sleep -Seconds 30
    }
}
