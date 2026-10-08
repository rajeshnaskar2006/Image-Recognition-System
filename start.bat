@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat once before starting the app.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py
set "app_result=%errorlevel%"
if not "%app_result%"=="0" (
    echo.
    echo The app stopped with an error. Run setup.bat if dependencies are missing.
    pause
)
exit /b %app_result%
