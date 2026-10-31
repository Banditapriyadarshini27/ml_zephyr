@echo off
REM ==============================================================================
REM Sketch2Shape AI - Automated Setup Script (Windows)
REM TechZephyr 4.0 Hackathon (IIT Bhubaneswar)
REM ==============================================================================

echo ========================================================
echo [Sketch2Shape AI] Initializing Windows Environment Setup...
echo ========================================================

if not exist outputs mkdir outputs
if not exist examples mkdir examples

REM Check for virtual environment
if not exist venv (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing locked dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ========================================================
echo [OK] Setup Complete!
echo Run the web application using: python app.py
echo ========================================================
pause
