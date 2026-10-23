
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sales_forecast_api.joblib"
DATA_PATH = ROOT_DIR / "data" / "processed" / "daily_sales_features.csv"


def load_artifact():
    artifact = joblib.load(MODEL_PATH)

    assert isinstance(artifact, dict), "Expected a model artifact dictionary"
    assert "model" in artifact, "Artifact is missing the trained model"
    assert "features" in artifact, "Artifact is missing the feature list"

    return artifact


def prepare_features(artifact):
    df = pd.read_csv(DATA_PATH)
    features = artifact["features"]

    missing_columns = set(features) - set(df.columns)
    assert not missing_columns, (
        f"Dataset is missing model features: {sorted(missing_columns)}"
    )

    X = df[features].tail(10)

    assert not X.empty, "No rows available for prediction"
    assert X.notna().all().all(), "Model features contain missing values"

    return artifact["model"], X


def test_model_predictions_are_finite():
    artifact = load_artifact()
    model, X = prepare_features(artifact)

    predictions = np.asarray(model.predict(X), dtype=float)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all(), (
        "Model returned NaN or infinite predictions"
    )


def test_model_predictions_are_non_negative():
    artifact = load_artifact()
    model, X = prepare_features(artifact)

    predictions = np.asarray(model.predict(X), dtype=float)

    assert (predictions >= 0).all(), (
        "Model returned negative sales predictions"
    )
