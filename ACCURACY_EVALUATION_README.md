# RFP Model Accuracy Evaluation System

This system provides comprehensive accuracy metrics for your RFP classification model. It evaluates the model's performance using historical decisions stored in `rfp_db.csv`.

## Available Evaluation Tools

### 1. Python Script (`evaluate_accuracy.py`)
The main evaluation script with multiple options:

```bash
# Full evaluation with multiple thresholds
python evaluate_accuracy.py

# Evaluate with specific threshold
python evaluate_accuracy.py --threshold 0.6

# Quick evaluation (less verbose output)
python evaluate_accuracy.py --threshold 0.5 --quick

# Show only model information
python evaluate_accuracy.py --info-only
```

### 2. PowerShell Script (`evaluate.ps1`)
Easy-to-use PowerShell interface:

```powershell
# Full evaluation
.\evaluate.ps1

# Specific threshold
.\evaluate.ps1 -Action threshold -Threshold 0.6

# Quick evaluation
.\evaluate.ps1 -Action quick

# Model information only
.\evaluate.ps1 -Action info
```

### 3. Batch File (`run_accuracy_evaluation.bat`)
Double-click to run a comprehensive evaluation.

## Metrics Provided

### Overall Metrics
- **Overall Accuracy**: Percentage of correct predictions
- **Correct Predictions**: Number of correct predictions out of total
- **Probability MAE**: Mean Absolute Error of probability predictions

### Confusion Matrix
Shows the breakdown of:
- True Positives (Approved correctly predicted as Approved)
- False Positives (Denied incorrectly predicted as Approved)
- False Negatives (Approved incorrectly predicted as Denied)
- True Negatives (Denied correctly predicted as Denied)

### Per-Class Metrics
For both "Approved" and "Denied" classes:
- **Precision**: Of all items predicted as this class, how many were correct?
- **Recall**: Of all actual items of this class, how many were correctly predicted?
- **F1-Score**: Harmonic mean of precision and recall

### Detailed Predictions
Individual prediction breakdown showing:
- Probability assigned by the model
- Predicted class
- Actual class
- Whether the prediction was correct

## Current Model Performance

Based on your current data:
- **Fine-tuned model**: Available
- **Historical decisions**: 23 total (19 Approved, 4 Denied)
- **Best threshold**: 0.5
- **Overall accuracy**: 100.00%

## Understanding the Results

### High Accuracy (90%+)
- Your model is performing excellently
- Consider deploying with confidence
- Continue monitoring with new data

### Medium Accuracy (70-90%)
- Model is reasonably good but has room for improvement
- Consider collecting more training data
- May need threshold adjustment

### Low Accuracy (<70%)
- Model needs significant improvement
- Collect more diverse training data
- Consider feature engineering improvements
- May need model architecture changes

## Threshold Selection

The system tests multiple thresholds (0.3 to 0.7) and recommends the best one. The threshold determines the probability cutoff for classification:

- **Lower threshold (0.3-0.4)**: More items classified as "Approved"
- **Higher threshold (0.6-0.7)**: More items classified as "Denied"
- **Balanced threshold (0.5)**: Equal treatment of both classes

## Data Requirements

For reliable evaluation, you need:
- Minimum 4 historical decisions
- At least 2 examples each of "Approved" and "Denied"
- Meaningful document summaries (>20 characters)

## Troubleshooting

### "Insufficient data for evaluation"
- Add more historical decisions to `rfp_db.csv`
- Ensure summaries are substantial (not just titles)

### Python import errors
- Make sure you're using the virtual environment
- Run from the correct directory

### Model loading issues
- Check if the fine-tuned model exists in `fine_tuned_longformer/`
- Verify all required packages are installed

## Command Examples

```bash
# Quick check of current model status
python evaluate_accuracy.py --info-only

# Fast accuracy check
python evaluate_accuracy.py --threshold 0.5 --quick

# Comprehensive analysis
python evaluate_accuracy.py

# Test different threshold
python evaluate_accuracy.py --threshold 0.6
```

The evaluation system will help you understand how well your RFP classification model is performing and guide you in making improvements.
