from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerFeatures(BaseModel):
    """Validated customer attributes accepted by the prediction API."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "credit_score": 650,
                "age": 42,
                "tenure": 5,
                "balance": 125000.0,
                "estimated_salary": 85000.0,
                "geography": "Germany",
                "gender": "Female",
            }
        }
    )

    credit_score: int = Field(..., ge=300, le=900)
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0, le=10)
    balance: float = Field(..., ge=0)
    estimated_salary: float = Field(..., ge=0)
    geography: Literal["France", "Germany", "Spain"]
    gender: Literal["Female", "Male"]

    @field_validator("balance", "estimated_salary")
    @classmethod
    def reasonable_money_value(cls, value: float) -> float:
        if value > 10_000_000:
            raise ValueError("value is outside the supported range")
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    prediction_label: Literal["Stayed", "Exited"]
    churn_probability: float
    model_version: str
    request_id: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
