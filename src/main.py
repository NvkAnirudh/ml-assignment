"""
Main entry point for the insurance enrollment prediction pipeline.

This script orchestrates data loading, preprocessing, model training,
and evaluation.
"""

import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import load_data, explore_data, preprocess_data
from model import train_models, evaluate_model, get_feature_importance


def main():
    """Run the complete ML pipeline."""
    print("=" * 60)
    print("Insurance Enrollment Prediction Pipeline")
    print("=" * 60)

    # Define data path
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "employee_data.csv"
    )

    # TODO: Implement pipeline steps
    # 1. Load data
    # 2. Explore data
    # 3. Preprocess data
    # 4. Train models
    # 5. Evaluate models
    # 6. Analyze feature importance

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
