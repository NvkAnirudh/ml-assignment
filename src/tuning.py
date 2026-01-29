"""
Hyperparameter tuning module using RandomizedSearchCV.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# Parameter grids
PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [0.62]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced']
    }
}


def get_base_models():
    """Return base models for tuning."""
    return {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        'SVM': SVC(probability=True, random_state=42)
    }


def tune_model(model_name, X_train, y_train, n_iter=20, cv=5):
    """
    Tune a single model using RandomizedSearchCV.

    Returns: best_model, best_params, best_score
    """
    from sklearn.metrics import f1_score
    import itertools
    import random

    base_models = get_base_models()
    param_grid = PARAM_GRIDS[model_name]

    # Manual randomized search for XGBoost (sklearn compatibility issue)
    if model_name == 'XGBoost':
        best_score = 0
        best_params = None
        best_model = None

        # Generate random parameter combinations
        param_keys = list(param_grid.keys())
        all_combinations = list(itertools.product(*param_grid.values()))
        random.seed(42)
        sampled = random.sample(all_combinations, min(n_iter, len(all_combinations)))

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for combo in sampled:
            params = dict(zip(param_keys, combo))
            scores = []

            for train_idx, val_idx in cv_splitter.split(X_train, y_train):
                model = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, **params)
                model.fit(X_train[train_idx], y_train.iloc[train_idx])
                y_pred = model.predict(X_train[val_idx])
                scores.append(f1_score(y_train.iloc[val_idx], y_pred))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, **params)
                best_model.fit(X_train, y_train)

        return best_model, best_params, best_score

    # Standard RandomizedSearchCV for other models
    model = base_models[model_name]

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_all_models(X_train, y_train, n_iter=20):
    """Tune all models and return results."""
    results = []
    best_models = {}

    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    for model_name in PARAM_GRIDS.keys():
        print(f"\nTuning {model_name}...")

        best_model, best_params, best_score = tune_model(
            model_name, X_train, y_train, n_iter=n_iter
        )

        best_models[model_name] = best_model
        results.append({
            'Model': model_name,
            'Best CV F1': round(best_score, 4),
            'Best Params': best_params
        })

        print(f"  Best CV F1: {best_score:.4f}")
        print(f"  Best Params: {best_params}")

    return best_models, results


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from data_processing import load_data, preprocess_data
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "employee_data.csv")

    # Load and preprocess
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, _, feature_names = preprocess_data(df)

    # Tune models
    best_models, tuning_results = tune_all_models(X_train, y_train, n_iter=20)

    # Evaluate tuned models on test set
    print("\n" + "=" * 60)
    print("TUNED MODEL PERFORMANCE (Test Set)")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 56)

    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"{name:<20} {acc:<12.4f} {f1:<12.4f} {auc:<12.4f}")
