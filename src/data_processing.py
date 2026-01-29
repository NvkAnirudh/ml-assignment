"""
Data processing module for insurance enrollment prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath: str) -> pd.DataFrame:
    """Load the employee data from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the data for model training.

    Steps:
    1. Drop employee_id
    2. Separate features and target
    3. One-hot encode categoricals
    4. StandardScale numericals
    5. Train/test split (stratified)

    Returns:
        X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    df = df.copy()

    # Drop employee_id
    if 'employee_id' in df.columns:
        df = df.drop('employee_id', axis=1)

    # Separate features and target
    X = df.drop('enrolled', axis=1)
    y = df['enrolled']

    # Define column types
    numerical_cols = ['age', 'salary', 'tenure_years']
    categorical_cols = ['gender', 'marital_status', 'employment_type', 'region']
    binary_cols = ['has_dependents']

    # Binary encode has_dependents
    X['has_dependents'] = (X['has_dependents'] == 'Yes').astype(int)

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'  # keeps has_dependents
    )

    # Train/test split (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit preprocessor on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_features) + binary_cols

    print(f"Training set: {X_train_processed.shape[0]:,} samples")
    print(f"Test set: {X_test_processed.shape[0]:,} samples")
    print(f"Features: {X_train_processed.shape[1]}")
    print(f"\nClass distribution (train):")
    print(f"  Not Enrolled: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Enrolled: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


def get_preprocessor_pipeline():
    """
    Return a sklearn pipeline for preprocessing new data.
    Useful for inference on new samples.
    """
    numerical_cols = ['age', 'salary', 'tenure_years']
    categorical_cols = ['gender', 'marital_status', 'employment_type', 'region']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "employee_data.csv")

    print("=" * 50)
    print("Data Preprocessing")
    print("=" * 50)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    print(f"\nFeature names:\n{feature_names}")
