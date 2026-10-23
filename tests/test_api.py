
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_history_endpoint():
    response = client.get("/history?days=7")

    assert response.status_code == 200

    payload = response.json()
    assert "history" in payload
    assert len(payload["history"]) <= 7

    for row in payload["history"]:
        assert "date" in row
        assert "sales" in row
        assert isinstance(row["sales"], (int, float))


def test_model_metrics_endpoint():
    response = client.get("/model-metrics")

    assert response.status_code == 200

    payload = response.json()
    assert "metrics" in payload
    assert len(payload["metrics"]) > 0


def test_forecast_endpoint():
    response = client.post(
        "/forecast",
        json={"horizon": 7},
    )

    assert response.status_code == 200

    payload = response.json()

    # Accept the common response formats used by the API.
    if isinstance(payload, list):
        predictions = payload
    else:
        predictions = next(
            (
                payload[key]
                for key in ("forecast", "forecasts", "predictions", "data")
                if isinstance(payload.get(key), list)
            ),
            None,
        )

    assert predictions is not None
    assert len(predictions) == 7

    for prediction in predictions:
        assert "date" in prediction
        assert any(
            key in prediction
            for key in ("predicted_sales", "prediction", "sales", "yhat")
        )

def test_history_rejects_negative_days():
    response = client.get("/history?days=-1")
    assert response.status_code == 422


def test_history_rejects_days_above_limit():
    response = client.get("/history?days=5001")
    assert response.status_code == 422


def test_forecast_rejects_invalid_horizon():
    response = client.post(
        "/forecast",
        json={"horizon": 0},
    )
    assert response.status_code in (400, 422)


def test_forecast_rejects_missing_horizon():
    response = client.post("/forecast", json={})
    assert response.status_code in (400, 422)
