@echo off
echo ============================================
echo RFP Analyzer - Flask Web Application
echo ============================================
echo.

cd /d "%~dp0"

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask not found. Installing requirements...
    pip install -r requirements_flask.txt
    echo.
)

echo Starting RFP Analyzer Web Server...
echo.
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python web_app.py

pause
