import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "CreditScore": rng.integers(350, 850, 80),
            "Age": rng.integers(18, 80, 80),
            "Tenure": rng.integers(0, 11, 80),
            "Balance": rng.uniform(0, 200000, 80),
            "EstimatedSalary": rng.uniform(10000, 200000, 80),
            "Geography_Germany": rng.integers(0, 2, 80),
            "Geography_Spain": rng.integers(0, 2, 80),
            "Gender_Male": rng.integers(0, 2, 80),
        }
    )
    y = (X["Age"] > 50).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "features": FEATURE_COLUMNS,
                "test_f1": 0.50,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(metadata_path))

    from app.model_service import model_service

    model_service._model = None
    model_service._metadata = {}
    model_service.model_path = model_path
    model_service.metadata_path = metadata_path

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


VALID_PAYLOAD = {
    "credit_score": 650,
    "age": 42,
    "tenure": 5,
    "balance": 125000.0,
    "estimated_salary": 85000.0,
    "geography": "Germany",
    "gender": "Female",
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] in [0, 1]
    assert body["prediction_label"] in ["Stayed", "Exited"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_version"] == "test-model"
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_invalid_age_returns_clear_error(client):
    payload = {**VALID_PAYLOAD, "age": 12}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_batch_prediction(client):
    response = client.post(
        "/predict/batch",
        json={"customers": [VALID_PAYLOAD, VALID_PAYLOAD]},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_ui_is_available(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Churn Predictor" in response.text
