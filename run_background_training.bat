@echo off
title RFP Model Background Training
echo =====================================================
echo    RFP MODEL BACKGROUND TRAINING
echo =====================================================
echo.
echo This will start model training in the background.
echo You can minimize this window and continue working.
echo Training will continue even if you switch tabs.
echo.
echo Press any key to start training, or close window to cancel...
pause > nul

echo.
echo Starting Python background training...
echo.

python background_training.py

echo.
echo Training session completed.
echo Press any key to close this window...
pause > nul
