@echo off
echo ===================================================
echo   YOLOv8 Car Counter Pro - Setup ^& Run
echo ===================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH. 
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b
)

echo.
echo [1/3] Setting up Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/3] Installing dependencies (this may take a few minutes)...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul
pip install -r requirements.txt

echo.
echo [3/3] Starting the Server...
echo Press Ctrl+C to stop the server at any time.
echo.
python -m app.main

pause
