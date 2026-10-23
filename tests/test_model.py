
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "sales_forecast_api.joblib"

FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "daily_sales_features.csv"
)


def test_trained_model_exists():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"


def test_trained_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_processed_dataset_exists():
    assert FEATURES_PATH.exists(), (
        f"Processed dataset not found: {FEATURES_PATH}"
    )


def test_processed_dataset_has_required_columns():
    df = pd.read_csv(FEATURES_PATH)

    required_columns = {"date", "sales"}

    assert required_columns.issubset(set(df.columns)), (
        f"Missing columns: {required_columns - set(df.columns)}"
    )


def test_processed_dataset_has_valid_dates():
    df = pd.read_csv(FEATURES_PATH)

    dates = pd.to_datetime(df["date"], errors="coerce")

    assert dates.notna().all(), "Dataset contains invalid dates"


def test_sales_values_are_valid():
    df = pd.read_csv(FEATURES_PATH)

    sales = pd.to_numeric(df["sales"], errors="coerce")

    assert sales.notna().all(), "Sales contains missing or non-numeric values"
    assert np.isfinite(sales).all(), "Sales contains infinite values"
    assert (sales >= 0).all(), "Sales contains negative values"


def test_dataset_is_sorted_by_date():
    df = pd.read_csv(FEATURES_PATH)

    dates = pd.to_datetime(df["date"], errors="coerce")

    assert dates.is_monotonic_increasing, (
        "Dataset dates are not sorted chronologically"
    )


def test_dataset_has_no_duplicate_dates():
    df = pd.read_csv(FEATURES_PATH)

    dates = pd.to_datetime(df["date"], errors="coerce")

    assert not dates.duplicated().any(), (
        "Dataset contains duplicate dates"
    )
