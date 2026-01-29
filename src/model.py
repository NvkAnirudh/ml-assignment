"""
Model training and evaluation module for insurance enrollment prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def get_models():
    """Return dictionary of models to train."""
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=10,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=0.62,  # ratio of negative/positive classes
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'SVM': SVC(
            class_weight='balanced',
            kernel='rbf',
            probability=True,  # needed for ROC-AUC
            random_state=42
        )
    }
    return models


def train_models(X_train, y_train, X_test, y_test):
    """Train all models and return trained models with results."""
    models = get_models()
    results = []
    trained_models = {}

    print("Training models...\n")
    print("-" * 70)

    for name, model in models.items():
        print(f"Training {name}...")

        # Cross-validation (manual for XGBoost compatibility)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            cv_model = clone(model) if name != 'XGBoost' else get_models()[name]
            cv_model.fit(X_train[train_idx], y_train.iloc[train_idx])
            cv_pred = cv_model.predict(X_train[val_idx])
            cv_scores.append(f1_score(y_train.iloc[val_idx], cv_pred))
        print(f"  CV F1: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")

        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        results.append(metrics)

    print("-" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1', ascending=False).reset_index(drop=True)

    return trained_models, results_df


def evaluate_model(model, X_test, y_test):
    """Evaluate a single model and return detailed metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    return metrics, y_pred, y_prob


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from a trained model."""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


def print_results(results_df):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Format percentages
    display_df = results_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

    print(display_df.to_string(index=False))
    print("=" * 70)

    best_model = results_df.iloc[0]['Model']
    best_f1 = results_df.iloc[0]['F1']
    print(f"\nBest model by F1: {best_model} ({best_f1:.3f})")


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from data_processing import load_data, preprocess_data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "employee_data.csv")

    print("=" * 70)
    print("Model Training & Evaluation")
    print("=" * 70)

    # Load and preprocess
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    # Train models
    trained_models, results_df = train_models(X_train, y_train, X_test, y_test)

    # Print results
    print_results(results_df)

    # Feature importance for best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]

    print(f"\nFeature Importance ({best_model_name}):")
    importance_df = get_feature_importance(best_model, feature_names, best_model_name)
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))
