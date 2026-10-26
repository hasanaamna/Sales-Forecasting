
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

HORIZON = 7
N_WINDOWS = 12
WINDOW_GAP = 28

FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
]


def make_features(date, history, calendar_row):
    """Create features using past sales and date-known information."""
    row = {
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "year": date.year,
        "day_of_year": date.dayofyear,
        "is_weekend": int(date.dayofweek >= 5),
        "is_holiday": calendar_row["is_holiday"],
        # "promotion_count": calendar_row["promotion_count"],
        # "promotion_missing": calendar_row["promotion_missing"],
        "sales_lag_1": history.iloc[-1],
        "sales_lag_7": history.iloc[-7],
        "sales_lag_14": history.iloc[-14],
        "sales_lag_28": history.iloc[-28],
        "sales_rolling_mean_7": history.iloc[-7:].mean(),
        "sales_rolling_mean_28": history.iloc[-28:].mean(),
    }
    return pd.DataFrame([row], columns=FEATURES)


def metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    denominator = np.abs(actual).sum()

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_percent": (
            np.abs(actual - predicted).sum() / denominator * 100
            if denominator else np.nan
        ),
    }


def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"], index_col="date")
    df = df.sort_index()

    required = FEATURES + ["sales"]
    missing_columns = set(required) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    # Exclude early rows with unavailable lag/rolling values.
    df = df.dropna(subset=required)

    # Select 12 non-overlapping seven-day test windows in the
    # most recent portion of the series, working backwards.
    latest_start = len(df) - HORIZON
    starts = [
        latest_start - i * WINDOW_GAP
        for i in reversed(range(N_WINDOWS))
    ]
    starts = [s for s in starts if s >= 28]

    if not starts:
        raise ValueError("Not enough history for rolling evaluation.")

    all_predictions = []

    for window_number, start in enumerate(starts, start=1):
        train = df.iloc[:start].copy()
        test = df.iloc[start:start + HORIZON].copy()

        train = train.dropna(subset=FEATURES + ["sales"])
        if len(train) < 100 or len(test) != HORIZON:
            continue

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[FEATURES], train["sales"])

        history = df["sales"].iloc[:start].copy()
        recursive_predictions = []

        for date in test.index:
            calendar_row = df.loc[date]
            X = make_features(date, history, calendar_row)
            prediction = max(0.0, float(model.predict(X)[0]))

            recursive_predictions.append(prediction)
            history.loc[date] = prediction

        # Seasonal-naive baseline uses the value seven days earlier.
        baseline_predictions = df["sales"].iloc[
            start - 7:start
        ].to_numpy()

        for i, date in enumerate(test.index):
            all_predictions.append({
                "window": window_number,
                "date": date,
                "actual_sales": float(test["sales"].iloc[i]),
                "random_forest_prediction": recursive_predictions[i],
                "baseline_prediction": float(baseline_predictions[i]),
            })

    results = pd.DataFrame(all_predictions)
    if results.empty:
        raise ValueError("No evaluation windows could be scored.")

    actual = results["actual_sales"]

    summary = pd.DataFrame([
        {
            "Model": "Seasonal Naive",
            **metrics(actual, results["baseline_prediction"]),
        },
        {
            "Model": "Random Forest (recursive)",
            **metrics(actual, results["random_forest_prediction"]),
        },
    ])

    summary.to_csv(REPORT_DIR / "rolling_forecast_metrics.csv", index=False)
    results.to_csv(REPORT_DIR / "rolling_forecast_predictions.csv", index=False)

    print("\nROLLING SEVEN-DAY EVALUATION")
    print(f"Windows evaluated: {results['window'].nunique()}")
    print(f"Forecast days evaluated: {len(results)}")
    print(f"Evaluation dates: {results['date'].min().date()} to "
          f"{results['date'].max().date()}")
    print(summary.round(2).to_string(index=False))

    # Plot the final window
    last_window = results[results["window"] == results["window"].max()]

    plt.figure(figsize=(12, 5))
    plt.plot(last_window["date"], last_window["actual_sales"],
             marker="o", label="Actual")
    plt.plot(last_window["date"], last_window["baseline_prediction"],
             marker="o", label="Seasonal Naive")
    plt.plot(last_window["date"], last_window["random_forest_prediction"],
             marker="o", label="Random Forest")
    plt.title("Final Rolling Seven-Day Forecast Window")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rolling_forecast_comparison.png", dpi=150)
    plt.close()

    print("\nSaved rolling evaluation reports in:", REPORT_DIR)


if __name__ == "__main__":
    main()
