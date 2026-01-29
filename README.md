# Insurance Enrollment Prediction

A machine learning pipeline to predict whether employees will enroll in a voluntary insurance product based on demographic and employment data.

## Project Structure

```
ml-assignment/
├── data/
│   └── employee_data.csv       # Input dataset
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── model.py                # Model training and evaluation
│   └── main.py                 # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── report.md                   # Analysis report
```

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the complete pipeline:
```bash
python src/main.py
```

## Dataset

The dataset contains ~10,000 employee records with the following features:
- `employee_id`: Unique identifier
- `age`: Employee age
- `gender`: Male, Female, Other
- `marital_status`: Single, Married, Divorced
- `salary`: Annual salary
- `employment_type`: Full-time, Part-time, Contract
- `region`: West, Midwest, Northeast, South
- `has_dependents`: Yes/No
- `tenure_years`: Years at company
- `enrolled`: Target variable (1 = enrolled, 0 = not enrolled)

## Results

See `report.md` for detailed analysis and results.
