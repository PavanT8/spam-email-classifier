import pytest
from fastapi.testclient import TestClient
from src.main import app

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

def test_predict_spam():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Congratulations! You have won a $1000 Walmart gift card. Click here to claim your prize."})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert data["prediction"] == "spam"

def test_predict_ham():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Hey John, are we still meeting for lunch at 12 PM?"})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "ham"
