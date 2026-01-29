"""
Main entry point for the insurance enrollment prediction pipeline.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import load_data, preprocess_data
from model import train_models, print_results, get_feature_importance


def main():
    """Run the complete ML pipeline."""
    print("=" * 60)
    print("Insurance Enrollment Prediction Pipeline")
    print("=" * 60)

    # Data path
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "employee_data.csv")

    # Step 1: Load data
    print("\n[1/4] Loading data...")
    df = load_data(data_path)

    # Step 2: Preprocess
    print("\n[2/4] Preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    # Step 3: Train models
    print("\n[3/4] Training models...")
    trained_models, results_df = train_models(X_train, y_train, X_test, y_test)

    # Step 4: Results
    print("\n[4/4] Results")
    print_results(results_df)

    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    for model_name in ['Random Forest', 'XGBoost', 'Logistic Regression']:
        model = trained_models[model_name]
        importance_df = get_feature_importance(model, feature_names, model_name)
        if importance_df is not None:
            print(f"\n{model_name}:")
            print(importance_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)

    return trained_models, results_df, preprocessor, feature_names


if __name__ == "__main__":
    main()
