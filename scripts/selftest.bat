@echo off
rem One-click self-test for Fuse: install packages, start the database, make sure the
rem test database exists, then run every test (unit + integration).
rem The output is shown here and saved to logs\selftest.log.
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set "LOG=logs\selftest.log"
echo Fuse self-test, %date% %time%> "%LOG%"

echo [1/4] Installing packages (uv sync)...
call uv sync >> "%LOG%" 2>&1 || goto failed

echo [2/4] Starting the database (Docker)...
docker compose up -d db >> "%LOG%" 2>&1 || goto failed
set /a tries=0
:waitdb
docker compose exec -T db pg_isready -U fuse -d fuse >nul 2>&1 && goto dbready
set /a tries+=1
if %tries% geq 40 goto failed
timeout /t 2 /nobreak >nul
goto waitdb
:dbready

echo [3/4] Making sure the test database exists...
docker compose exec -T db psql -U fuse -d fuse -tAc "SELECT 1 FROM pg_database WHERE datname='fuse_test'" | findstr "1" >nul
if errorlevel 1 docker compose exec -T db psql -U fuse -d fuse -c "CREATE DATABASE fuse_test" >> "%LOG%" 2>&1
docker compose exec -T db psql -U fuse -d fuse_test -c "CREATE EXTENSION IF NOT EXISTS vector" >> "%LOG%" 2>&1

echo [4/4] Running all tests (about a minute)...
set "TEST_DATABASE_URL=postgresql://fuse:fuse@127.0.0.1:55432/fuse_test"
call uv run pytest -q -rs >> "%LOG%" 2>&1
set "RESULT=%errorlevel%"
echo.
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
echo.
if "%RESULT%"=="0" (echo SELF-TEST PASSED) else (echo SELF-TEST FAILED: see %LOG%)
echo SELFTEST_EXIT=%RESULT%>> "%LOG%"
pause
exit /b %RESULT%

:failed
echo.
echo SELF-TEST STOPPED EARLY: see %LOG%
echo SELFTEST_EXIT=setup-failed>> "%LOG%"
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
pause
exit /b 1
