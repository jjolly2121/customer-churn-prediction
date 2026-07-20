# Data Documentation

## Selected Dataset

This project uses a public bank customer churn dataset for supervised binary classification. The goal is to predict whether a customer exited the bank.

Source dataset family:

- Kaggle Churn Modelling dataset: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
- Kaggle customer churn variants with the same banking churn schema are also available, including: https://www.kaggle.com/datasets/moaazhammad/customer-churn-prediction

The dataset is appropriate for this capstone because it is structured, tabular, business-oriented, and directly aligned with a practical machine learning engineering use case: identifying customers at risk of churn so a bank can prioritize retention actions.

## Files In This Folder

```text
data/
├── Customer Churn new.csv
├── customer_churn_cleaned.csv
└── model_benchmark_results.csv
```

`Customer Churn new.csv` is the raw dataset used for data wrangling and exploration.

`customer_churn_cleaned.csv` is the cleaned modeling dataset exported from the data-wrangling notebook.

`model_benchmark_results.csv` contains benchmark model results from the initial model comparison workflow.

## Dataset Collection Process

The dataset was collected from a public dataset repository rather than through a custom data extraction process. The collection process for this capstone was:

1. Identify public churn datasets suitable for a banking churn prediction problem.
2. Review dataset descriptions, available columns, target definition, file format, and modeling relevance.
3. Select the bank customer churn dataset because it contains a clear binary target and customer-level account features.
4. Download the CSV file from the public dataset page.
5. Store the raw CSV in the project `data/` directory.
6. Load the raw CSV in the data-wrangling notebook.
7. Inspect schema, row count, data types, missing values, duplicate rows, and target distribution.
8. Remove fields that should not be used as predictors.
9. Encode categorical variables for model training.
10. Export the cleaned dataset for downstream modeling notebooks.

No web scraping or API extraction was required for this capstone dataset. In a production project, the equivalent collection process would normally involve extracting customer, account, and churn-label data from internal databases, CRM systems, event logs, or a data warehouse.

## Data Dictionary

| Column | Description | Used For Modeling |
|---|---|---|
| `RowNumber` | Row identifier from the source file | No |
| `CustomerId` | Customer identifier | No |
| `Surname` | Customer surname | No |
| `CreditScore` | Customer credit score | Yes |
| `Geography` | Customer country or region | Yes |
| `Gender` | Customer gender | Yes |
| `Age` | Customer age | Yes |
| `Tenure` | Number of years the customer has been with the bank | Yes |
| `Balance` | Customer account balance | Yes |
| `EstimatedSalary` | Estimated customer salary | Yes |
| `Exited` | Target variable: `1` means exited, `0` means stayed | Target |

The modeling pipeline removes `RowNumber`, `CustomerId`, and `Surname` because they are identifiers or personal-name fields rather than behavioral predictors. The final model uses one-hot encoded versions of `Geography` and `Gender`.

## Initial Data Quality Checks

The raw dataset contains:

- 10,000 rows
- 11 columns
- No missing values
- No duplicate rows found during inspection
- Binary target variable `Exited`
- Imbalanced churn distribution, with most customers staying and a smaller share exiting

Potential outliers were reviewed during exploration. Values such as high balances, older ages, or high salaries were retained because they can represent real customers and may contain useful churn signals.

## Real-World Data Collection Plan

If this project were implemented inside a bank, the data collection process would need to be planned before modeling. A production collection plan would include:

1. Identify source systems:
   - customer master records
   - account balances
   - product ownership
   - transaction summaries
   - CRM interactions
   - support tickets
   - account closure records

2. Define the churn label:
   - closed account
   - inactive for a defined period
   - transferred balance out
   - stopped using core products

3. Define the observation window:
   - collect predictor features from a fixed historical period
   - define churn over a future prediction window
   - prevent leakage by excluding future information from training features

4. Build an extraction process:
   - SQL queries or ETL jobs from a warehouse
   - scheduled batch exports
   - data validation checks
   - versioned dataset snapshots

5. Validate data quality:
   - missing values
   - duplicate customers
   - invalid ranges
   - inconsistent categories
   - changes in feature distributions
   - target leakage risks

6. Address privacy and governance:
   - remove direct identifiers where possible
   - restrict access to sensitive customer data
   - document data lineage
   - follow internal compliance requirements

7. Prepare for ongoing updates:
   - refresh data on a fixed schedule
   - monitor drift
   - retrain the model when performance declines or data changes materially

This capstone uses a static public CSV, but the same planning steps would apply before building a production churn prediction system with real customer data.

## Notes On Scope

The public dataset is smaller than a typical production banking dataset. This project compensates by focusing on the machine learning engineering workflow: data cleaning, model comparison, artifact persistence, API serving, Docker packaging, testing, and cloud deployment.
