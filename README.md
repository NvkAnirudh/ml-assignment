# Insurance Enrollment Prediction

A machine learning pipeline to predict whether employees will enroll in a voluntary insurance product based on demographic and employment data.

## Quick Start

```bash
# Clone and navigate
cd ml-assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/main.py
```

## Project Structure

```
ml-assignment/
├── data/
│   └── employee_data.csv       # Input dataset (10,000 rows)
├── models/
│   ├── best_model.pkl          # Trained model (Random Forest)
│   └── preprocessor.pkl        # Fitted preprocessor
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── model.py                # Model training and evaluation
│   ├── tuning.py               # Hyperparameter tuning
│   ├── experiment.py           # MLflow experiment tracking
│   ├── api.py                  # FastAPI REST endpoints
│   └── main.py                 # Main entry point
├── mlruns/                     # MLflow experiment logs
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── report.md                   # Detailed analysis report
```

## Dataset

| Feature | Description |
|---------|-------------|
| `employee_id` | Unique identifier |
| `age` | Employee age (22-64) |
| `gender` | Male, Female, Other |
| `marital_status` | Single, Married, Divorced, Widowed |
| `salary` | Annual salary ($2k-$120k) |
| `employment_type` | Full-time, Part-time, Contract |
| `region` | West, Midwest, Northeast, South |
| `has_dependents` | Yes/No |
| `tenure_years` | Years at company (0-36) |
| `enrolled` | **Target** (1 = enrolled, 0 = not enrolled) |

## Models

Five classification models were trained and evaluated:

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Decision Tree | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 |
| XGBoost | 1.000 | 1.000 | 1.000 |
| SVM | 0.969 | 0.975 | 0.997 |
| Logistic Regression | 0.894 | 0.912 | 0.971 |

## Key Findings

1. **Top predictors:** `has_dependents`, `employment_type`, `salary`, `age`.
2. **Useless features:** `gender`, `region`, `marital_status`, `tenure_years`.
3. **Tree models achieve perfect accuracy** because there are deterministic patterns in the synthetic data.

## Usage

**Run full pipeline:**
```bash
python src/main.py
```

**Run individual modules:**
```bash
# Data preprocessing only
python src/data_processing.py

# Model training only
python src/model.py

# Hyperparameter tuning
python src/tuning.py

# MLflow experiment tracking
python src/experiment.py
```

**Explore data interactively:**
```bash
jupyter notebook notebooks/eda.ipynb
```

**Run REST API:**
```bash
uvicorn src.api:app --reload
# Visit http://localhost:8000/docs for Swagger UI
```

**View MLflow experiments:**
```bash
mlflow ui
# Visit http://localhost:5000
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

See `requirements.txt` for full list.

## Report

See [report.md](report.md) for detailed analysis including:
- Data observations
- Preprocessing steps
- Model selection rationale
- Evaluation results
- Feature importance
- Next steps
