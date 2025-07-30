# RFP Model Accuracy Evaluation PowerShell Script
# This script provides easy commands to evaluate your RFP model accuracy

param(
    [string]$Action = "full",
    [double]$Threshold = 0.5,
    [switch]$Quick,
    [switch]$Info
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "evaluate_accuracy.py"

# Configure Python environment
$env:PYTHONPATH = $scriptDir

Write-Host "RFP Model Accuracy Evaluation" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

switch ($Action.ToLower()) {
    "full" {
        Write-Host "Running full accuracy evaluation..." -ForegroundColor Yellow
        python $pythonScript
    }
    "threshold" {
        Write-Host "Running evaluation with threshold $Threshold..." -ForegroundColor Yellow
        if ($Quick) {
            python $pythonScript --threshold $Threshold --quick
        } else {
            python $pythonScript --threshold $Threshold
        }
    }
    "info" {
        Write-Host "Showing model information..." -ForegroundColor Yellow
        python $pythonScript --info-only
    }
    "quick" {
        Write-Host "Running quick evaluation..." -ForegroundColor Yellow
        python $pythonScript --threshold $Threshold --quick
    }
    default {
        Write-Host "Usage:" -ForegroundColor Cyan
        Write-Host "  .\evaluate.ps1                    # Full evaluation"
        Write-Host "  .\evaluate.ps1 -Action threshold -Threshold 0.6  # Specific threshold"
        Write-Host "  .\evaluate.ps1 -Action quick      # Quick evaluation"
        Write-Host "  .\evaluate.ps1 -Action info       # Model info only"
        Write-Host "  .\evaluate.ps1 -Action threshold -Threshold 0.7 -Quick  # Quick with threshold"
    }
}
