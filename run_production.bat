@echo off
title RFP Analyzer - Production Mode
echo =====================================================
echo    RFP ANALYZER - PRODUCTION FLASK SERVER
echo =====================================================
echo.
echo 🚀 Launching production server (no auto-reload)...
echo 💡 This prevents interruptions during processing
echo.

cd /d "%~dp0"

python run_production.py

echo.
echo Press any key to exit...
pause > nul
