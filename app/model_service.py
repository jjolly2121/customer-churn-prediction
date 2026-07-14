import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Iterable

import joblib
import pandas as pd

from .schemas import CustomerFeatures

logger = logging.getLogger(__name__)

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


class ModelService:
    """Thread-safe loader and prediction interface for the churn model."""

    def __init__(self) -> None:
        self._model = None
        self._metadata: dict = {}
        self._lock = Lock()

        project_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(
            os.getenv(
                "MODEL_PATH",
                project_root / "artifacts" / "final_random_forest.joblib",
            )
        )
        self.metadata_path = Path(
            os.getenv(
                "MODEL_METADATA_PATH",
                project_root / "artifacts" / "model_metadata.json",
            )
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        return str(
            self._metadata.get(
                "model_version",
                self._metadata.get("model_name", "baseline-random-forest-v1"),
            )
        )

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    def load(self) -> None:
        """Load the model and metadata once at application startup."""
        with self._lock:
            if self.loaded:
                return

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found at {self.model_path}"
                )

            self._model = joblib.load(self.model_path)

            if self.metadata_path.exists():
                with self.metadata_path.open("r", encoding="utf-8") as file:
                    self._metadata = json.load(file)
            else:
                self._metadata = {
                    "model_name": "baseline-random-forest-v1",
                    "features": FEATURE_COLUMNS,
                }

            expected = self._metadata.get("features", FEATURE_COLUMNS)
            if list(expected) != FEATURE_COLUMNS:
                raise ValueError(
                    "Model metadata feature order does not match the API schema."
                )

            logger.info(
                "Model loaded",
                extra={"model_version": self.model_version},
            )

    @staticmethod
    def to_frame(customers: Iterable[CustomerFeatures]) -> pd.DataFrame:
        """Convert validated API inputs into the training feature schema."""
        records = []
        for customer in customers:
            records.append(
                {
                    "CreditScore": customer.credit_score,
                    "Age": customer.age,
                    "Tenure": customer.tenure,
                    "Balance": customer.balance,
                    "EstimatedSalary": customer.estimated_salary,
                    "Geography_Germany": int(customer.geography == "Germany"),
                    "Geography_Spain": int(customer.geography == "Spain"),
                    "Gender_Male": int(customer.gender == "Male"),
                }
            )

        return pd.DataFrame(records, columns=FEATURE_COLUMNS)

    def predict(self, customers: list[CustomerFeatures]) -> list[dict]:
        if not self.loaded:
            self.load()

        frame = self.to_frame(customers)
        predictions = self._model.predict(frame)
        probabilities = self._model.predict_proba(frame)[:, 1]

        return [
            {
                "prediction": int(prediction),
                "prediction_label": "Exited" if int(prediction) == 1 else "Stayed",
                "churn_probability": round(float(probability), 6),
            }
            for prediction, probability in zip(predictions, probabilities)
        ]


model_service = ModelService()
