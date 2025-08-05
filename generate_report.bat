@echo off
echo Starting HTML Report Generator...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Install required packages if needed
echo Installing/checking required packages...
pip install -q pandas scikit-learn transformers torch

REM Generate HTML report
echo Generating HTML report...
python generate_html_report.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ HTML report generated successfully!
    echo 📁 Look for 'rfp_analyzer_report.html' in this folder
    echo 🌐 Double-click the HTML file to open it in your browser
) else (
    echo.
    echo ❌ Error generating report
)

echo.
pause
