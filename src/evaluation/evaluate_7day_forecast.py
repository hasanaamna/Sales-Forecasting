
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "random_forest_sales.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"

HORIZON = 7


def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"], index_col="date")
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    features = saved["features"]

    df = df.sort_index()

    # Match the historical validation cutoff used earlier.
    validation_start = df.index.max() - pd.Timedelta(days=89)
    train = df.loc[df.index < validation_start].copy()
    actual = df.loc[df.index >= validation_start, "sales"].copy()

    # Fit a fresh model on training-period observations only.
    train = train.dropna(subset=features + ["sales"])
    model.fit(train[features], train["sales"])

    # Start with the observed training history.
    history = df.loc[df.index < validation_start, "sales"].copy()

    # Recursive seven-day forecast.
    forecast_dates = pd.date_range(
        validation_start,
        periods=HORIZON,
        freq="D",
    )

    predictions = []

    for date in forecast_dates:
        weekday = date.dayofweek

        # Build features for this forecast date.
        row = {
            "day_of_week": weekday,
            "day_of_month": date.day,
            "month": date.month,
            "year": date.year,
            "day_of_year": date.dayofyear,
            "is_weekend": int(weekday >= 5),
            "sales_lag_1": history.iloc[-1],
            "sales_lag_7": history.iloc[-7],
            "sales_lag_14": history.iloc[-14],
            "sales_lag_28": history.iloc[-28],
            "sales_rolling_mean_7": history.iloc[-7:].mean(),
            "sales_rolling_mean_28": history.iloc[-28:].mean(),
        }

        # Other features must be known at forecast time.
        # For this first isolated test, use historical weekday
        # averages for promotion/holiday/transaction features.
        for feature in features:
            if feature not in row:
                row[feature] = float(
                    train.groupby("day_of_week")[feature]
                    .mean()
                    .get(weekday, train[feature].mean())
                )

        X = pd.DataFrame([row], columns=features)
        prediction = max(0.0, float(model.predict(X)[0]))

        predictions.append(prediction)
        history.loc[date] = prediction

    actual_7 = actual.reindex(forecast_dates)

    if actual_7.isna().any():
        raise ValueError("Actual sales are unavailable for forecast dates.")

    errors = actual_7.to_numpy() - np.array(predictions)

    mae = mean_absolute_error(actual_7, predictions)
    rmse = np.sqrt(mean_squared_error(actual_7, predictions))
    wape = np.abs(errors).sum() / np.abs(actual_7.to_numpy()).sum() * 100

    print("\nSEVEN-DAY RECURSIVE FORECAST")
    print(f"Forecast start: {forecast_dates.min().date()}")
    print(f"Forecast end:   {forecast_dates.max().date()}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"WAPE: {wape:.2f}%")

    results = pd.DataFrame({
        "date": forecast_dates,
        "actual_sales": actual_7.to_numpy(),
        "predicted_sales": predictions,
    })

    results.to_csv(REPORT_DIR / "seven_day_forecast.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(results["date"], results["actual_sales"], marker="o", label="Actual")
    plt.plot(results["date"], results["predicted_sales"], marker="o", label="Forecast")
    plt.title("Seven-Day Sales Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "seven_day_forecast.png", dpi=150)
    plt.close()

    print(f"\nSaved: {REPORT_DIR / 'seven_day_forecast.csv'}")
    print(f"Saved: {REPORT_DIR / 'seven_day_forecast.png'}")


if __name__ == "__main__":
    main()
