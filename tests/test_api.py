import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["docs_url"] == "/docs"

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data

def test_sentiment_endpoint():
    response = client.get("/api/v1/stocks/AAPL/sentiment")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "summary" in data
    assert "articles" in data
    assert data["summary"]["total_articles"] > 0

def test_stock_history_endpoint():
    response = client.get("/api/v1/stocks/AAPL/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert len(data["data"]) == 10
    assert "close" in data["data"][0]

def test_predict_endpoint():
    payload = {
        "symbol": "AAPL",
        "time_step": 15,
        "epochs": 1,
        "use_sentiment": True
    }
    response = client.post("/api/v1/stocks/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "forecast_30_days" in data
    assert len(data["forecast_30_days"]) == 30
    assert "directional_accuracy" in data
    assert "judging_score" in data

def test_prediction_history_endpoint():
    response = client.get("/api/v1/predictions/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

