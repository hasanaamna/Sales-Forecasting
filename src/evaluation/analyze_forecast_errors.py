
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "rolling_forecast_predictions.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"])

    # Determine each forecast's position within its seven-day window.
    df = df.sort_values(["window", "date"]).copy()
    df["forecast_horizon"] = df.groupby("window").cumcount() + 1

    # Calculate absolute errors for each model.
    df["rf_abs_error"] = (
        df["actual_sales"] - df["random_forest_prediction"]
    ).abs()

    df["baseline_abs_error"] = (
        df["actual_sales"] - df["baseline_prediction"]
    ).abs()

    # Evaluate errors by forecast horizon.
    rows = []

    for horizon, group in df.groupby("forecast_horizon"):
        actual_total = group["actual_sales"].abs().sum()

        rows.append({
            "forecast_horizon": horizon,
            "observations": len(group),
            "RF_MAE": group["rf_abs_error"].mean(),
            "Baseline_MAE": group["baseline_abs_error"].mean(),
            "RF_WAPE_percent": (
                group["rf_abs_error"].sum() / actual_total * 100
                if actual_total else np.nan
            ),
            "Baseline_WAPE_percent": (
                group["baseline_abs_error"].sum() / actual_total * 100
                if actual_total else np.nan
            ),
        })

    horizon_report = pd.DataFrame(rows)
    horizon_report.to_csv(
        REPORT_DIR / "forecast_horizon_analysis.csv",
        index=False,
    )

    print("\nERROR BY FORECAST HORIZON")
    print(horizon_report.round(2).to_string(index=False))

    print("\nSaved:")
    print(REPORT_DIR / "forecast_horizon_analysis.csv")


if __name__ == "__main__":
    main()
