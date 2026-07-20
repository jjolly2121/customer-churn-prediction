# Customer Churn Prediction API

A machine learning engineering capstone project that predicts whether a bank customer is likely to exit using demographic and financial account features. The project includes the full modeling workflow, saved model artifacts, a FastAPI prediction service, automated API tests, and Docker deployment configuration.

## Project Overview

Customer churn prediction helps organizations identify customers who may leave so retention efforts can be targeted before revenue is lost. This project uses a 10,000-record bank customer churn dataset and predicts the `Exited` target:

- `0` = customer stayed
- `1` = customer exited

Because only about 20% of customers exited, the primary model-selection metric is F1 score for the churn class. ROC-AUC is used as a secondary measure of class separation.

## Project Workflow

1. Survey existing churn-prediction research and reproduce a baseline workflow.
2. Clean and explore the raw dataset.
3. Create a processed modeling dataset.
4. Benchmark Logistic Regression, Decision Tree, and Random Forest models.
5. Compare candidate models using five-fold stratified cross-validation.
6. Tune the strongest candidates with `RandomizedSearchCV`.
7. Evaluate the selected model on an untouched test set.
8. Scale and validate the prototype.
9. Package the trained model behind a production-style FastAPI service.
10. Test and containerize the application for deployment.

## Repository Structure

```text
customer-churn-prediction/
├── app/
│   ├── main.py
│   ├── model_service.py
│   ├── schemas.py
│   └── logging_config.py
├── artifacts/
│   ├── final_random_forest.joblib
│   ├── model_metadata.json
│   └── model and scaling result files
├── data/
│   ├── Customer Churn new.csv
│   ├── customer_churn_cleaned.csv
│   └── README.md
├── docs/
│   ├── Step_9_Deployment_Method_and_Engineering_Plan.docx
│   └── 10_Deployment_Architecture.docx
├── notebooks/
│   ├── 01_research_reproduction.ipynb
│   ├── 02_data_wrangling_eda_executed.ipynb
│   ├── 03_benchmark_models_executed.ipynb
│   ├── 04_model_experiments.ipynb
│   ├── 05_scale_prototype.ipynb
│   └── 06_api_demo.ipynb
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-deployment.txt
└── README.md
```

## Models Evaluated

The modeling workflow compared:

- Dummy Classifier
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Histogram Gradient Boosting

The baseline Random Forest achieved the strongest balance of F1 score, recall, and deployment simplicity, so it was retained as the final model.

## Final Model Performance

Performance on the untouched 20% test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.7660 |
| Precision | 0.4452 |
| Recall | 0.6093 |
| F1 Score | 0.5145 |
| ROC-AUC | 0.7838 |

The strongest Random Forest predictors were age, balance, estimated salary, and credit score. Feature importance describes predictive influence and should not be interpreted as proof of causation.

## Application Features

- FastAPI prediction API
- Browser-based prediction form at `/`
- Swagger documentation at `/docs`
- Single and batch prediction endpoints
- Health and model-information endpoints
- Structured request and prediction logging
- Input validation and controlled error responses
- Docker container configuration
- Automated API tests

## Required Model Files

The application expects these existing project artifacts:

```text
artifacts/final_random_forest.joblib
artifacts/model_metadata.json
```

## Run Locally With Python

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-deployment.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

- User interface: `http://127.0.0.1:8000`
- Swagger API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Model information: `http://127.0.0.1:8000/model-info`

## Run With Docker

```bash
docker build -t customer-churn-api .
docker run --rm -p 8000:8000 customer-churn-api
```

Or:

```bash
docker-compose up --build
```

## Run Tests

```bash
python -m pytest -q
```

## Example API Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 650,
    "age": 42,
    "tenure": 5,
    "balance": 125000,
    "estimated_salary": 85000,
    "geography": "Germany",
    "gender": "Female"
  }'
```

Example response:

```json
{
  "prediction": 1,
  "prediction_label": "Exited",
  "churn_probability": 0.61,
  "model_version": "Baseline Random Forest",
  "request_id": "generated-request-id"
}
```

The exact prediction and probability depend on the trained model artifact.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Browser prediction interface |
| GET | `/docs` | Interactive Swagger documentation |
| GET | `/health` | Application and model health |
| GET | `/model-info` | Model version, features, and stored metrics |
| POST | `/predict` | Single-customer prediction |
| POST | `/predict/batch` | Batch prediction for up to 1,000 customers |

## Logging And Monitoring

The application writes structured JSON logs to standard output. Each request receives a request ID, and logs include endpoint, HTTP method, response status, latency, model version, prediction class, and exceptions when failures occur.

Direct customer identifiers are not accepted by the API and therefore are not written to logs.

## Limitations

- The model misses approximately 39% of customers who churn.
- Precision is limited, so retention outreach would include false positives.
- The dataset contains a relatively small set of customer attributes.
- Feature importance does not establish causal relationships.
- A production deployment should add external monitoring, drift detection, and a retraining process.

## Future Improvements

Potential extensions include threshold optimization based on retention costs, probability calibration, additional behavioral features, fairness testing, drift monitoring, and deployment to a managed cloud service.

## Author

Jeffrey Jolly
