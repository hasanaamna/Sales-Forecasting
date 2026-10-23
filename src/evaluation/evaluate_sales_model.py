
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "validation_predictions.csv"
)
MODEL_FILE = PROJECT_ROOT / "models" / "random_forest_sales.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"])
    saved_model = joblib.load(MODEL_FILE)
    model = saved_model["model"]
    features = saved_model["features"]

    # --------------------------------------------------------
    # 1. Actual vs predicted sales
    # --------------------------------------------------------
    plt.figure(figsize=(15, 6))
    plt.plot(
        df["date"], df["actual_sales"],
        label="Actual Sales", linewidth=1.8
    )
    plt.plot(
        df["date"], df["random_forest_prediction"],
        label="Random Forest Prediction", linewidth=1.5
    )
    plt.title("Actual vs Predicted Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "actual_vs_predicted.png",
        dpi=150
    )
    plt.close()

    # --------------------------------------------------------
    # 2. Prediction errors
    # --------------------------------------------------------
    df["error"] = (
        df["actual_sales"] - df["random_forest_prediction"]
    )

    plt.figure(figsize=(15, 5))
    plt.plot(df["date"], df["error"], linewidth=1)
    plt.axhline(0, linestyle="--")
    plt.title("Daily Forecast Errors")
    plt.xlabel("Date")
    plt.ylabel("Actual - Predicted Sales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "forecast_errors.png",
        dpi=150
    )
    plt.close()

    # --------------------------------------------------------
    # 3. Feature importance
    # --------------------------------------------------------
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values()

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        header=["importance"]
    )

    plt.figure(figsize=(10, 7))
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "feature_importance.png",
        dpi=150
    )
    plt.close()

    print("Evaluation completed.")
    print("\nTop 10 features:")
    print(importance.sort_values(ascending=False).head(10).round(4))

    print("\nGenerated:")
    print(REPORT_DIR / "actual_vs_predicted.png")
    print(REPORT_DIR / "forecast_errors.png")
    print(REPORT_DIR / "feature_importance.png")


if __name__ == "__main__":
    main()
