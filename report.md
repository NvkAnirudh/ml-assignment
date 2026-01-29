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

<!-- TODO: Document model selection reasoning -->

## 3. Evaluation Results

<!-- TODO: Add performance metrics and comparisons -->

## 4. Key Takeaways

<!-- TODO: Summarize main findings -->

## 5. Next Steps

<!-- TODO: What would be done with more time -->
