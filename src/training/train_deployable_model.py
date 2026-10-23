
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

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


def main():
    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
        index_col="date",
    ).sort_index()

    # Keep only features available from the calendar and
    # previously observed sales. Do not use future transactions
    # or future promotion counts as predictors.
    df = df.dropna(subset=FEATURES + ["sales"])

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(df[FEATURES], df["sales"])

    artifact = {
        "model": model,
        "features": FEATURES,
        "trained_through": str(df.index.max().date()),
        "model_type": "RandomForestRegressor",
    }

    output = MODEL_DIR / "sales_forecast_api.joblib"
    joblib.dump(artifact, output)

    print("Deployable model trained and saved.")
    print(f"Training observations: {len(df):,}")
    print(f"Training through: {df.index.max().date()}")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
