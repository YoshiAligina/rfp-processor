@echo off
echo ============================================
echo RFP Model Accuracy Evaluation
echo ============================================
echo.

cd /d "%~dp0"

echo Running comprehensive accuracy evaluation...
echo.

"C:/Users/Yoshita.X.Aligina/OneDrive - Quest Diagnostics/Desktop/rfp-processor/.venv/Scripts/python.exe" evaluate_accuracy.py

echo.
echo Evaluation complete. Press any key to exit...
pause >nul
