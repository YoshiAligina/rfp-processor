@echo off
title RFP Document Processing Monitor
echo =====================================================
echo    RFP DOCUMENT PROCESSING MONITOR
echo =====================================================
echo.
echo This monitor shows real-time document processing status.
echo You can run this while uploading documents to see progress.
echo.
echo Press any key to start monitoring, or close window to cancel...
pause > nul

echo.
echo Starting processing monitor...
echo.

python monitor_processing.py

echo.
echo Monitoring session ended.
echo Press any key to close this window...
pause > nul
