# Customer Churn Prediction API

FastAPI application for predicting whether a bank customer is likely to exit.

## Application features

- FastAPI prediction API
- Browser-based prediction form at `/`
- Swagger documentation at `/docs`
- Single and batch prediction endpoints
- Health and model-information endpoints
- Structured request and prediction logging
- Input validation and controlled error responses
- Docker container configuration
- Automated API tests

## Required model files

The application expects these existing project artifacts:

```text
artifacts/final_random_forest.joblib
artifacts/model_metadata.json
```

## Run locally with Python

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

## Run with Docker

```bash
docker build -t customer-churn-api .
docker run --rm -p 8000:8000 customer-churn-api
```

Or:

```bash
docker-compose up --build
```

## Run tests

```bash
python -m pytest -q
```

## Example API request

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

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Browser prediction interface |
| GET | `/docs` | Interactive Swagger documentation |
| GET | `/health` | Application and model health |
| GET | `/model-info` | Model version, features, and stored metrics |
| POST | `/predict` | Single-customer prediction |
| POST | `/predict/batch` | Batch prediction for up to 1,000 customers |

## Logging and monitoring

The application writes structured JSON logs to standard output. Each request receives a request ID, and logs include endpoint, HTTP method, response status, latency, model version, prediction class, and exceptions when failures occur.

Direct customer identifiers are not accepted by the API and therefore are not written to logs.
