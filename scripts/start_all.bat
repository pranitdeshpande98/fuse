@echo off
rem Starts everything Fuse needs, each in its own window:
rem   the database (Docker), Job Alerts (if its folder is found), and Fuse (bot + worker).
rem Close a window (or press Ctrl+C in it) to stop that part.
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs

echo Starting the database...
docker compose up -d db

rem Job Alerts: JOB_ALERTS_DIR if set, else the usual Desktop folder.
if "%JOB_ALERTS_DIR%"=="" set "JOB_ALERTS_DIR=%USERPROFILE%\Desktop\autojobalert"
if exist "%JOB_ALERTS_DIR%\start.bat" (
    echo Starting Job Alerts from %JOB_ALERTS_DIR%...
    start "Job Alerts" /D "%JOB_ALERTS_DIR%" cmd /k start.bat
) else (
    echo Job Alerts not found at %JOB_ALERTS_DIR%: skipping it.
)

echo Starting Fuse (bot + worker); its log is also saved to logs\fuse.log...
start "Fuse" cmd /k "uv run fuse 2>&1 | powershell -NoProfile -Command \"$input | Tee-Object -FilePath logs\fuse.log\""
echo Done. You can close this window.
timeout /t 5 >nul
