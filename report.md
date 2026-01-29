# Insurance Enrollment Prediction - Analysis Report

## 1. Data Observations

The dataset contains 10,000 employees with 9 features and a binary target indicating whether each person enrolled in the voluntary insurance product. There are no missing values, and the target distribution is slightly imbalanced: 61.7% of employees enroll (6,174 people), while 38.3% do not (3,826 people). This imbalance is noticeable but mild enough that it can be handled with class weights rather than aggressive resampling.

When we look at which groups actually enroll, a clear story emerges:

- Employees **with dependents** are far more likely to enroll than those without. Roughly 8 out of 10 employees with dependents enroll, compared with about 3 to 4 out of 10 employees without dependents.
- **Employment type** also matters. Most full-time employees choose to enroll (about three quarters of them), whereas enrollment among part-time and contract workers is closer to one in three.
- **Salary** and **age** provide additional, but secondary, signals. Enrolled employees tend to earn more (around \$69k on average versus \$58k for those who do not enroll) and are somewhat older (mid 40s compared with late 30s).
- **Tenure** and **region** do not show meaningful differences. Employees across regions enroll at very similar rates, and years at the company is not a strong differentiator once other factors are taken into account.

Put simply, the employees most likely to enroll are full time workers with dependents, typically older and better paid. Where they live and how long they have been at the company adds little predictive value.

## 2. Data Preprocessing

The preprocessing pipeline is intentionally simple and mirrors how this data would be prepared in production.

- **Identifier removal:** The `employee_id` column is dropped because it is only an identifier and carries no information about enrollment.
- **Target and features:** The `enrolled` column is kept as the target, and all remaining columns are treated as input features.
- **Binary encoding:** The `has_dependents` column is converted from `"Yes"/"No"` to 1/0. This keeps the meaning clear while making it directly usable by the models.
- **Categorical encoding:** Categorical features (`gender`, `marital_status`, `employment_type`, `region`) are one hot encoded with one category dropped per feature. The dropped categories act as reference groups and prevent redundant columns.
- **Numeric scaling:** Continuous variables (`age`, `salary`, `tenure_years`) are standardized so that models see them on comparable scales. This helps algorithms that are sensitive to feature magnitude and makes optimization more stable.
- **Train/test split and imbalance handling:** The data is split into an 80/20 train–test split, stratified on the target to preserve the original class ratio. Because the imbalance is modest (about 62:38), models use `class_weight='balanced'` instead of synthetic oversampling.

After preprocessing, the model works with 14 features: 3 scaled numerical features, 10 one hot encoded categorical features, and 1 binary indicator for dependents.

## 3. Model Choices & Rationale

Several complementary models were trained to understand both performance and behavior:

- **Logistic Regression** serves as a strong, interpretable baseline. Its coefficients make it easy to see which features increase or decrease the likelihood of enrollment.
- A **Decision Tree** captures simple, human readable rules such as “full time with dependents” versus “part time with no dependents,” and exposes clear thresholds.
- A **Random Forest** combines many decision trees to reduce variance and usually produces more stable predictions than any single tree.
- **XGBoost**, a boosted tree ensemble, is included because it typically performs extremely well on tabular, structured data like this.
- An **SVM** offers a different perspective, using margins and decision boundaries rather than tree‑based rules, and provides a useful comparison across algorithm families.

All models are configured with `class_weight='balanced'` to counter the moderate class imbalance without changing the underlying data.

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

1. **Dependents + Employment type are the primary drivers.** Employees with dependents in full-time roles almost always enroll.
2. **Tree-based models achieve perfect accuracy.** The data has clear, learnable decision boundaries.
3. **Logistic Regression underperforms.** It cannot capture the non-linear threshold effects (age plateau at 30, salary jump at 60k).
4. **Some features are useless.** Gender, region, marital status, and tenure add no predictive value.

## 6. Next Steps

- **Model explainability:** use SHAP values to explain individual predictions.
- **Real-world validation:** test the approach on actual (non-synthetic) data with more noise.
- **Monitoring:** track model drift and prediction distributions over time.

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

**Observation:** Tuning improved SVM significantly (F1: 0.975 to 0.988). Tree models were already near-perfect.

## 8. Experiment Tracking (MLflow)

Used MLflow to log experiments for reproducibility and comparison.

For each experiment, MLflow records:

- **Parameters**: all relevant hyperparameters for each model, so that any run can be exactly reproduced.
- **Metrics**: standard classification metrics, including accuracy, precision, recall, F1, and ROC-AUC, logged for easy comparison across runs.
- **Artifacts**: confusion matrix images, ROC curves, and any other diagnostic plots generated during evaluation.
- **Models**: serialized versions of the trained models suitable for later deployment or further analysis.

The following artifacts are stored for day to day use:

- `models/best_model.pkl`: the best performing model (a Random Forest in this case).
- `models/preprocessor.pkl`: the fitted preprocessing pipeline, needed to transform new data at inference time.
- `mlruns/`: the complete MLflow experiment logs, which can be explored using the MLflow UI.

**View experiments:** Run `mlflow ui` in project directory to launch dashboard.

## 9. REST API (FastAPI)

Built a REST API for real-time predictions.

The FastAPI service exposes a small set of endpoints:

- `GET /` provides a simple health check and indicates whether the model has been loaded.
- `GET /model/info` returns basic metadata about the deployed model and the features it expects.
- `POST /predict` accepts a single employee record and returns a prediction, the associated probability, and a human readable label.
- `POST /predict/batch` accepts a list of employees and returns predictions for each one in a single call.

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
