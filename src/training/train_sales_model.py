
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_sales_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================
VALIDATION_DAYS = 90

FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "promotion_count",
    "promotion_missing",
    "transactions_lag_1",
    "transactions_lag_7",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
]


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    denominator = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / denominator * 100
        if denominator != 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_FILE}"
        )

    print("Loading prepared sales data...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
        index_col="date",
    ).sort_index()

    # Remove early rows where lag/rolling features are unavailable.
    df = df.dropna(subset=FEATURES + ["sales"])

    if len(df) <= VALIDATION_DAYS:
        raise ValueError("Not enough observations for validation.")

    # Chronological split: last 90 days are held out.
    split_position = len(df) - VALIDATION_DAYS

    train = df.iloc[:split_position].copy()
    valid = df.iloc[split_position:].copy()

    X_train = train[FEATURES]
    y_train = train["sales"]

    X_valid = valid[FEATURES]
    y_valid = valid["sales"]

    print(f"\nTraining observations: {len(train):,}")
    print(f"Validation observations: {len(valid):,}")
    print(f"Training ends: {train.index.max().date()}")
    print(f"Validation starts: {valid.index.min().date()}")
    print(f"Validation ends: {valid.index.max().date()}")

    # ========================================================
    # MODEL 1: SEASONAL-NAIVE BASELINE
    # Predict sales using the same weekday from the previous week.
    # ========================================================
    baseline_predictions = valid["sales_lag_7"].to_numpy()

    # ========================================================
    # MODEL 2: RANDOM FOREST
    # One-day-ahead evaluation using previously observed sales.
    # ========================================================
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    rf_predictions = model.predict(X_valid)

    # ========================================================
    # EVALUATION
    # ========================================================
    results = pd.DataFrame(
        [
            {
                "Model": "Seasonal Naive (7-day lag)",
                **calculate_metrics(y_valid, baseline_predictions),
            },
            {
                "Model": "Random Forest",
                **calculate_metrics(y_valid, rf_predictions),
            },
        ]
    )

    print("\nVALIDATION RESULTS")
    print(results.round(2).to_string(index=False))

    # Save metrics
    results.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # Save actual and predicted values
    predictions = pd.DataFrame(
        {
            "date": valid.index,
            "actual_sales": y_valid.to_numpy(),
            "baseline_prediction": baseline_predictions,
            "random_forest_prediction": rf_predictions,
        }
    )

    predictions.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False,
    )

    # Save trained model and feature order
    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_DIR / "random_forest_sales.joblib",
    )

    print("\nSaved files:")
    print(f"Metrics: {REPORT_DIR / 'model_comparison.csv'}")
    print(f"Predictions: {REPORT_DIR / 'validation_predictions.csv'}")
    print(f"Model: {MODEL_DIR / 'random_forest_sales.joblib'}")

    print(
        "\nNote: This is a rolling one-day-ahead evaluation, "
        "not a full seven-day-ahead forecast."
    )


if __name__ == "__main__":
    main()
