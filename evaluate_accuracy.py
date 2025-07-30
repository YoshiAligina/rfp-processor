#!/usr/bin/env python3
"""
RFP Model Accuracy Evaluation Script

This script provides a command-line interface to evaluate the accuracy of the RFP classification model.
It uses the historical decisions in rfp_db.csv to test the model's performance.

Usage:
    python evaluate_accuracy.py                    # Run full evaluation
    python evaluate_accuracy.py --threshold 0.6    # Use specific threshold
    python evaluate_accuracy.py --quick            # Quick evaluation (less verbose)
"""

import argparse
import sys
import os

# Add the current directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import evaluate_model_accuracy, run_accuracy_evaluation, get_model_info

def main():
    parser = argparse.ArgumentParser(description='Evaluate RFP Model Accuracy')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Probability threshold for classification (default: test multiple)')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Proportion of data to use for testing (default: 0.3)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick evaluation with minimal output')
    parser.add_argument('--info-only', action='store_true',
                        help='Show model information only')
    
    args = parser.parse_args()
    
    try:
        if args.info_only:
            # Just show model information
            model_info = get_model_info()
            print("RFP Model Information:")
            print(f"  Fine-tuned model available: {model_info['has_fine_tuned_model']}")
            print(f"  Historical decisions: {model_info['historical_decisions']}")
            print(f"  Approved: {model_info['approved_count']}, Denied: {model_info['denied_count']}")
            print(f"  Model path: {model_info['model_path']}")
            return
        
        if args.threshold is not None:
            # Evaluate with specific threshold
            print(f"Evaluating model accuracy with threshold {args.threshold}...")
            result = evaluate_model_accuracy(
                threshold=args.threshold, 
                test_size=args.test_size,
                verbose=not args.quick
            )
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                sys.exit(1)
            
            if args.quick:
                print(f"Accuracy: {result['overall_accuracy']:.2%}")
                print(f"Correct: {result['correct_predictions']}/{result['total_predictions']}")
        else:
            # Run comprehensive evaluation
            run_accuracy_evaluation()
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
