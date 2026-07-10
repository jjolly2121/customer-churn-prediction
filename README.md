# Customer Churn Prediction

A machine learning capstone project that predicts whether a bank customer will exit using demographic and financial account features.

## Project Overview

Customer churn prediction helps organizations identify customers who may leave so that retention efforts can be targeted before revenue is lost. This project develops a reproducible classification workflow using a 10,000-record bank customer churn dataset.

The target variable is `Exited`:

- `0` = customer stayed
- `1` = customer exited

Because only about 20% of customers exited, the primary model-selection metric is the F1 score for the churn class. ROC-AUC is used as a secondary measure of class separation.

## Project Workflow

The repository documents the project from research through final model selection:

1. Survey existing churn-prediction research and reproduce a baseline workflow.
2. Clean and explore the raw dataset.
3. Create a processed modeling dataset.
4. Benchmark Logistic Regression, Decision Tree, and Random Forest models.
5. Compare seven candidate models using five-fold stratified cross-validation.
6. Tune the strongest candidates with `RandomizedSearchCV`.
7. Evaluate the selected model on an untouched test set.
8. Save and validate the trained model and supporting result artifacts.

## Repository Structure

```text
customer-churn-prediction/
├── artifacts/
│   ├── candidate_comparison.csv
│   ├── cross_validation_results.csv
│   ├── feature_importance.csv
│   ├── final_random_forest.joblib
│   ├── final_test_metrics.csv
│   └── model_metadata.json
├── data/
│   ├── Customer Churn new.csv
│   └── customer_churn_cleaned.csv
├── docs/
│   └── research_summary.md
├── notebooks/
│   ├── 01_research_reproduction.ipynb
│   ├── 02_data_wrangling_eda_executed.ipynb
│   ├── 03_benchmark_models_executed.ipynb
│   └── 04_model_experiments.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Python 3.12 was used for development.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/jjolly2121/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Run the notebooks in numerical order. The notebooks include path checks so they can be launched from either the repository root or the `notebooks/` directory.

## Models Evaluated

The final experiment compared:

- Dummy Classifier
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Histogram Gradient Boosting

The baseline Random Forest achieved the highest mean cross-validation F1 score and generalized better than the tuned Random Forest, so it was retained as the final model.

## Final Model Performance

Performance on the untouched 20% test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.7660 |
| Precision | 0.4452 |
| Recall | 0.6093 |
| F1 Score | 0.5145 |
| ROC-AUC | 0.7838 |
| Average Precision | ~0.51 |

The final model correctly identified 248 of 407 customers who exited. It missed 159 churners and produced 309 false-positive retention alerts.

## Feature Importance

The strongest Random Forest predictors were:

1. Age
2. Balance
3. Estimated Salary
4. Credit Score

Feature importance describes predictive influence and should not be interpreted as proof of causation.

## Saved Artifacts

The `artifacts/` directory contains:

- The compressed final Random Forest model
- Cross-validation results
- Baseline-versus-tuned model comparisons
- Final test metrics
- Feature-importance values
- Model metadata and parameters

The compressed model is approximately 3.65 MB. It was reloaded successfully, and its predictions matched the original in-memory model.

## Limitations

- The model misses approximately 39% of customers who churn.
- Precision is limited, so retention outreach would include false positives.
- The dataset contains a relatively small set of customer attributes.
- Feature importance does not establish causal relationships.
- The current repository is a modeling prototype and does not yet include a production API or user interface.

## Future Improvements

Potential extensions include threshold optimization based on retention costs, probability calibration, additional behavioral features, fairness testing, drift monitoring, and deployment through a lightweight web application or REST API.

## Author

Jeffrey Jolly
