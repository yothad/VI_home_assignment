# WellCo Churn Prediction Assignment

## Project Overview
This repository contains a churn Xgboost model for predicting member churn.


## Data Sources
All data files are in the `data/` directory:

- **churn_labels.csv**: Member IDs, signup dates, churn labels, and outreach flags
- **claims.csv**: Member ICD-10 diagnosis codes and dates
- **app_usage.csv**: App session events per member
- **web_visits.csv**: Web page visits, titles, and descriptions per member

See the `schema_*.md` files in `data/` for detailed column descriptions.

## Pipeline Steps
The main pipeline is modularized in the `steps/` directory:

1. **Import** (`imports.py`): Loads all raw CSV data into pandas DataFrames.
2. **Preprocess** (`preprocess.py`):
	- Pivots claims to binary ICD features
	- Aggregates app usage counts
	- Pivots web visits to binary page features
	- Calculates subscription length
	- Merges all features into a single DataFrame
3. **Model** (`model.py`):
	- Splits data into train/test
	- Applies SMOTE for class balance
	- Selects top features with RFE (XGBoost)
	- Trains XGBoost classifier
	- Tunes threshold for best F1 score
	- Evaluates and reports metrics
4. **Export** (`export.py`):
	- Scores live (non-churned, non-outreached) members
	- Exports high-risk members to `export/exported_members.csv`

## How to Run
You can run the pipeline step-by-step in a Jupyter notebook (see `run.ipynb` or `analysis.ipynb`), or by importing and executing the classes in the `steps/` directory.

## Output
The main output is `export/exported_members.csv`, listing member IDs and churn risk scores for high-risk, live members.

## Comments
Notebooks are used for simplicity of the home assignment, it's easy to see step-by=step-run.
Also, for deployment I would have used a yml settings file.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- shap
- plotly
- jupyterlab
- jupyter_contrib_nbextensions

All requirements are listed in `requirements.txt`.

### How to install requirements

Open a terminal in the project root and run:

```sh
pip install -r requirements.txt
```
