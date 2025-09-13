@echo off
title Node.js Risk Simulation Server
echo Starting Node.js Risk Simulation Server...
echo.

cd /d "C:\Users\annuy\OneDrive\Documents\projects\Monte-Carlo-Simulation\backend\node"

:restart
echo [%date% %time%] Starting Node.js server...
node --expose-gc index-fixed.js

echo.
echo [%date% %time%] Server stopped with exit code: %ERRORLEVEL%

if %ERRORLEVEL% NEQ 0 (
    echo Server crashed! Restarting in 3 seconds...
    timeout /t 3 /nobreak >nul
    goto restart
) else (
    echo Server shut down normally. Press any key to restart or close this window.
    pause
    goto restart
)
