import logging
import time
import uuid
from contextlib import asynccontextmanager
from html import escape

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from .logging_config import configure_logging
from .model_service import model_service
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load()
    except Exception:
        logger.exception("Application startup failed while loading model")
        raise
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts whether a bank customer is likely to exit. "
        "Interactive API documentation is available at /docs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "Unhandled request failure",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "latency_ms": latency_ms,
            },
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning(
        "Request validation failed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": 422,
        },
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "details": exc.errors(),
            "request_id": request_id,
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.loaded else "degraded",
        model_loaded=model_service.loaded,
        model_version=model_service.model_version,
    )


@app.get("/model-info", tags=["System"])
def model_info() -> dict:
    return {
        "model_version": model_service.model_version,
        "features": model_service.metadata.get("features", []),
        "test_metrics": {
            key: value
            for key, value in model_service.metadata.items()
            if key.startswith("test_")
        },
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
)
def predict(
    customer: CustomerFeatures,
    request: Request,
) -> PredictionResponse:
    request_id = request.state.request_id

    try:
        result = model_service.predict([customer])[0]
    except Exception as exc:
        logger.exception(
            "Prediction failed",
            extra={"request_id": request_id, "path": "/predict"},
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "request_id": request_id,
            },
        ) from exc

    logger.info(
        "Prediction completed",
        extra={
            "request_id": request_id,
            "model_version": model_service.model_version,
            "prediction": result["prediction"],
        },
    )

    return PredictionResponse(
        **result,
        model_version=model_service.model_version,
        request_id=request_id,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    request_id = request.state.request_id
    raw_results = model_service.predict(payload.customers)

    results = [
        PredictionResponse(
            **result,
            model_version=model_service.model_version,
            request_id=request_id,
        )
        for result in raw_results
    ]

    return BatchPredictionResponse(
        predictions=results,
        count=len(results),
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def user_interface() -> str:
    """Simple browser UI for manual prediction requests."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Customer Churn Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 18px; background: #f6f7f9; }
    .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    label { display: block; font-weight: 600; margin-bottom: 4px; }
    input, select { width: 100%; box-sizing: border-box; padding: 9px; }
    button { margin-top: 18px; padding: 11px 18px; cursor: pointer; }
    #result { margin-top: 20px; padding: 14px; border-radius: 8px; background: #eef2f6; white-space: pre-wrap; }
    .links { margin-bottom: 16px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Customer Churn Predictor</h1>
    <div class="links"><a href="/docs">Swagger API docs</a> | <a href="/health">Health check</a></div>
    <form id="prediction-form">
      <div class="grid">
        <div><label>Credit score</label><input id="credit_score" type="number" min="300" max="900" value="650" required /></div>
        <div><label>Age</label><input id="age" type="number" min="18" max="100" value="42" required /></div>
        <div><label>Tenure</label><input id="tenure" type="number" min="0" max="10" value="5" required /></div>
        <div><label>Balance</label><input id="balance" type="number" min="0" step="0.01" value="125000" required /></div>
        <div><label>Estimated salary</label><input id="estimated_salary" type="number" min="0" step="0.01" value="85000" required /></div>
        <div><label>Geography</label><select id="geography"><option>France</option><option selected>Germany</option><option>Spain</option></select></div>
        <div><label>Gender</label><select id="gender"><option>Female</option><option>Male</option></select></div>
      </div>
      <button type="submit">Predict churn</button>
    </form>
    <div id="result">Enter customer information and submit the form.</div>
  </div>

  <script>
    document.getElementById("prediction-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const result = document.getElementById("result");
      result.textContent = "Loading...";

      const payload = {
        credit_score: Number(document.getElementById("credit_score").value),
        age: Number(document.getElementById("age").value),
        tenure: Number(document.getElementById("tenure").value),
        balance: Number(document.getElementById("balance").value),
        estimated_salary: Number(document.getElementById("estimated_salary").value),
        geography: document.getElementById("geography").value,
        gender: document.getElementById("gender").value
      };

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          result.textContent = JSON.stringify(data, null, 2);
          return;
        }
        result.textContent =
          `Prediction: ${data.prediction_label}\n` +
          `Churn probability: ${(data.churn_probability * 100).toFixed(2)}%\n` +
          `Model: ${data.model_version}\n` +
          `Request ID: ${data.request_id}`;
      } catch (error) {
        result.textContent = `Request failed: ${error}`;
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
