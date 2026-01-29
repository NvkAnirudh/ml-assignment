# Insurance Enrollment Prediction - Analysis Report

## 1. Data Observations

**Dataset:** 10,000 employees with 9 features and 1 target variable. No missing values.

**Target Distribution:**
- Enrolled: 61.7% (6,174)
- Not Enrolled: 38.3% (3,826)

Slight class imbalance, but manageable without aggressive resampling.

**What drives enrollment?**

| Feature | Finding |
|---------|---------|
| **Dependents** | Strongest signal. 79.7% with dependents enroll vs 34.8% without |
| **Employment Type** | Full-time: 75.3% enroll. Part-time/Contract: ~30% |
| **Salary** | Higher earners enroll more (avg $69k vs $58k for non-enrolled) |
| **Age** | Older employees enroll more (avg 45.6 vs 38.8 years) |
| **Tenure** | No meaningful correlation with enrollment |
| **Region** | All regions ~61-63% — not predictive |

**Bottom line:** Employees with dependents and full-time status are the primary enrollers. Salary and age provide secondary signals. Region and tenure don't matter much.

## 2. Data Preprocessing

| Step | What | Why |
|------|------|-----|
| Drop `employee_id` | Removed | Not predictive |
| Binary encode | `has_dependents` → 0/1 | Already binary |
| One-hot encode | `gender`, `marital_status`, `employment_type`, `region` | No ordinal relationship between categories |
| Drop first category | Female, Divorced, Contract, Midwest | Avoid multicollinearity; these become reference categories |
| Scale numericals | StandardScaler on `age`, `salary`, `tenure_years` | Handles outliers better than MinMaxScaler |
| Train/test split | 80/20, stratified | Preserves class ratio in both sets |
| Class imbalance | Using `class_weight='balanced'` in models | Mild imbalance (62:38) doesn't need SMOTE |

**Final feature count:** 14 (3 numerical + 10 one-hot encoded + 1 binary)

## 3. Model Choices & Rationale

| Model | Why Included |
|-------|--------------|
| **Logistic Regression** | Interpretable baseline; shows feature coefficients |
| **Decision Tree** | Simple, explainable; captures non-linear thresholds |
| **Random Forest** | Robust ensemble (bagging); reduces variance |
| **XGBoost** | Boosting ensemble; often best for tabular data |
| **SVM** | Margin-based approach; different algorithm family |

All models used `class_weight='balanced'` to handle the mild 62:38 class imbalance.

## 4. Evaluation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Decision Tree | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| XGBoost | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| SVM | 0.969 | 0.986 | 0.964 | 0.975 | 0.997 |
| Logistic Regression | 0.894 | 0.935 | 0.891 | 0.912 | 0.971 |

**Why perfect scores?** The synthetic data contains deterministic rules:
- Part-time/Contract + No dependents → 0% enrollment (1,179 employees, none enrolled)
- Full-time + Dependents → 92.8% enrollment

Tree models recover these rules exactly. This is not overfitting — train/test gap is 0.000.

**Feature Importance (Random Forest):**

| Feature | Importance |
|---------|------------|
| has_dependents | 31.5% |
| employment_type_Full-time | 27.2% |
| salary | 22.7% |
| age | 18.3% |
| Others | ~0% |

Four features drive nearly 100% of predictions. Gender, marital status, region, and tenure have no predictive value.

## 5. Key Takeaways

1. **Dependents + Employment type are the primary drivers** — employees with dependents in full-time roles almost always enroll
2. **Tree-based models achieve perfect accuracy** — the data has clear, learnable decision boundaries
3. **Logistic Regression underperforms** — can't capture the non-linear threshold effects (age plateau at 30, salary jump at 60k)
4. **Some features are useless** — gender, region, marital status, tenure add no predictive value

## 6. Next Steps

- **Model explainability** — SHAP values for individual prediction explanations
- **Real-world validation** — test on actual (non-synthetic) data with more noise
- **Monitoring** — track model drift and prediction distributions over time

---

# Bonus Work

## 7. Hyperparameter Tuning

Used `RandomizedSearchCV` with 5-fold stratified CV to find optimal parameters.

**Best Parameters Found:**

| Model | Best Params | CV F1 |
|-------|-------------|-------|
| Random Forest | `n_estimators=300, max_depth=5, min_samples_split=5, min_samples_leaf=2` | 0.9999 |
| XGBoost | `n_estimators=100, max_depth=7, learning_rate=0.05, subsample=0.8` | 0.9997 |
| SVM | `C=100, kernel=rbf, gamma=scale` | 0.9760 |

**Tuned Model Performance (Test Set):**

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Random Forest | 0.9995 | 0.9996 | 1.0000 |
| XGBoost | 0.9995 | 0.9996 | 1.0000 |
| SVM | 0.9855 | 0.9883 | 0.9988 |

**Observation:** Tuning improved SVM significantly (F1: 0.975 → 0.988). Tree models were already near-perfect.

## 8. Experiment Tracking (MLflow)

Used MLflow to log experiments for reproducibility and comparison.

**What's tracked:**

| Item | Description |
|------|-------------|
| Parameters | All model hyperparameters |
| Metrics | accuracy, precision, recall, f1, roc_auc |
| Artifacts | Confusion matrix plots, ROC curves, trained models |
| Models | Serialized sklearn models for deployment |

**Saved artifacts:**
- `models/best_model.pkl` — Best performing model (Random Forest)
- `models/preprocessor.pkl` — Fitted preprocessor for inference
- `mlruns/` — MLflow experiment logs

**View experiments:** Run `mlflow ui` in project directory to launch dashboard.

## 9. REST API (FastAPI)

Built a REST API for real-time predictions.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

**Example request:**
```json
{
  "age": 45,
  "gender": "Male",
  "marital_status": "Married",
  "salary": 75000,
  "employment_type": "Full-time",
  "region": "West",
  "has_dependents": "Yes",
  "tenure_years": 5.0
}
```

**Example response:**
```json
{
  "prediction": 1,
  "probability": 0.9388,
  "label": "Enrolled"
}
```

**Run API:** `uvicorn src.api:app --reload`

**Docs:** Visit `http://localhost:8000/docs` for interactive Swagger UI.
