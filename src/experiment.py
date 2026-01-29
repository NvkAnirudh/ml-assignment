"""
MLflow experiment tracking for model training and evaluation.
"""

import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import load_data, preprocess_data
from tuning import tune_all_models


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Not Enrolled', 'Enrolled'])
    ax.set_yticklabels(['Not Enrolled', 'Enrolled'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=16, fontweight='bold')

    plt.colorbar(im)
    plt.tight_layout()

    path = f'/tmp/cm_{model_name.replace(" ", "_")}.png'
    plt.savefig(path)
    plt.close()
    return path


def plot_roc_curve(y_true, y_prob, model_name):
    """Create and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend()
    plt.tight_layout()

    path = f'/tmp/roc_{model_name.replace(" ", "_")}.png'
    plt.savefig(path)
    plt.close()
    return path


def run_experiment(experiment_name="insurance_enrollment"):
    """Run full experiment with MLflow tracking."""

    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "employee_data.csv")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)

    # Load and preprocess
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    # Tune models
    best_models, tuning_results = tune_all_models(X_train, y_train, n_iter=15)

    # Track each model
    best_overall_f1 = 0
    best_overall_model = None
    best_overall_name = None

    for name, model in best_models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nLogging {name}...")

            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }

            # Log parameters
            params = model.get_params()
            for key, value in params.items():
                if value is not None and not callable(value):
                    try:
                        mlflow.log_param(key, value)
                    except:
                        pass

            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log plots
            cm_path = plot_confusion_matrix(y_test, y_pred, name)
            roc_path = plot_roc_curve(y_test, y_prob, name)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

            # Track best model
            if metrics['f1'] > best_overall_f1:
                best_overall_f1 = metrics['f1']
                best_overall_model = model
                best_overall_name = name

    # Save best model
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_overall_model, best_model_path)
    print(f"\nBest model ({best_overall_name}) saved to: {best_model_path}")

    # Save preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")

    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETE")
    print(f"Best Model: {best_overall_name} (F1: {best_overall_f1:.4f})")
    print(f"View results: mlflow ui")
    print("=" * 60)

    return best_overall_model, preprocessor, feature_names


if __name__ == "__main__":
    run_experiment()
