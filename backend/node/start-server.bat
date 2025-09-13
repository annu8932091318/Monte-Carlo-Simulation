@echo off
echo Starting Node.js server with enhanced monitoring...
echo.

REM Kill any existing Node.js processes on port 5010
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5010"') do (
    echo Killing existing process on port 5010: %%a
    taskkill /f /pid %%a 2>nul
)

echo.
echo Starting server with logging...
echo.

REM Start the server with enhanced logging and auto-restart
:start_server
node --expose-gc index-fixed.js
echo.
echo Server stopped with exit code: %ERRORLEVEL%
echo.

if %ERRORLEVEL% NEQ 0 (
    echo Server crashed! Exit code: %ERRORLEVEL%
    echo Waiting 5 seconds before restart...
    timeout /t 5 /nobreak >nul
    echo Restarting server...
    goto start_server
) else (
    echo Server shut down normally.
)

pause
