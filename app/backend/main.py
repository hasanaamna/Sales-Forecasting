
from datetime import date, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "sales_forecast_api.joblib"

app = FastAPI(
    title="Sales Forecasting API",
    description="API for forecasting daily retail sales.",
    version="1.0.0",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_FILE = (
    PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
)

METRICS_FILE = PROJECT_ROOT / "reports" / "model_comparison.csv"


@app.get("/history")
def get_history(days: int = Query(default=0, ge=0, le=5000)):
    if not HISTORY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {HISTORY_FILE}",
        )

    df = pd.read_csv(HISTORY_FILE)

    required = {"date", "sales"}
    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail=f"Expected columns {required}; found {list(df.columns)}",
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"]).sort_values("date")

    if days:
        df = df.tail(days)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return {
        "history": df[["date", "sales"]].to_dict(orient="records"),
        "count": len(df),
    }



@app.get("/model-metrics")
def get_model_metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {METRICS_FILE}",
        )

    df = pd.read_csv(METRICS_FILE)

    # Normalize column names for the frontend.
    rename_map = {
        "Model": "model",
        "WAPE_percent": "WAPE",
    }
    df = df.rename(columns=rename_map)

    return {
        "metrics": df.where(pd.notna(df), None).to_dict(orient="records")
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    horizon: int = Field(default=7, ge=1, le=7)


def load_resources():
    if not MODEL_FILE.exists():
        raise RuntimeError(
            "Deployable model not found. Run "
            "src/training/train_deployable_model.py first."
        )

    if not DATA_FILE.exists():
        raise RuntimeError("Processed sales data not found.")

    artifact = joblib.load(MODEL_FILE)
    history = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
        index_col="date",
    ).sort_index()

    return artifact, history


def make_row(forecast_date, sales_history, historical_features, features):
    weekday = forecast_date.dayofweek

    row = {
        "day_of_week": weekday,
        "day_of_month": forecast_date.day,
        "month": forecast_date.month,
        "year": forecast_date.year,
        "day_of_year": forecast_date.dayofyear,
        "is_weekend": int(weekday >= 5),
        "sales_lag_1": sales_history.iloc[-1],
        "sales_lag_7": sales_history.iloc[-7],
        "sales_lag_14": sales_history.iloc[-14],
        "sales_lag_28": sales_history.iloc[-28],
        "sales_rolling_mean_7": sales_history.iloc[-7:].mean(),
        "sales_rolling_mean_28": sales_history.iloc[-28:].mean(),
    }

    # Holiday information is sourced from the historical calendar
    # where available. Future dates without a known holiday entry
    # default to a non-holiday; this is a limitation of this version.
    if "is_holiday" in features:
        if forecast_date in historical_features.index:
            row["is_holiday"] = int(
                historical_features.loc[forecast_date, "is_holiday"]
            )
        else:
            row["is_holiday"] = 0

    return pd.DataFrame([row], columns=features)


@app.get("/")
def root():
    return {
        "message": "Sales Forecasting API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_exists": MODEL_FILE.exists(),
        "data_exists": DATA_FILE.exists(),
    }

class ForecastRequest(BaseModel):
    horizon: int = Field(..., ge=1, le=7)


@app.post("/forecast")
def forecast(request: ForecastRequest):
    horizon = request.horizon
    try:
        artifact, df = load_resources()
        model = artifact["model"]
        features = artifact["features"]

        if len(df) < 28:
            raise HTTPException(
                status_code=400,
                detail="At least 28 days of history are required.",
            )

        sales_history = df["sales"].copy()
        historical_features = df

        predictions = []
        last_date = sales_history.index.max()

        for step in range(1, request.horizon + 1):
            forecast_date = last_date + timedelta(days=step)

            row = make_row(
                forecast_date,
                sales_history,
                historical_features,
                features,
            )

            prediction = max(
                0.0,
                float(model.predict(row)[0]),
            )

            predictions.append({
                "date": forecast_date.date().isoformat(),
                "predicted_sales": round(prediction, 2),
            })

            # Use the prediction, not future actual sales,
            # to construct the next forecast's lag features.
            sales_history.loc[pd.Timestamp(forecast_date)] = prediction

        return {
            "forecast_horizon": request.horizon,
            "history_through": last_date.date().isoformat(),
            "forecast": predictions,
            "model": artifact.get("model_type", "Unknown"),
            "note": (
                "Exploratory forecast. Future holiday effects are "
                "simplified; validate before operational use."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {exc}",
        ) from exc
