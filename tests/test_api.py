import ml_api
from fastapi.testclient import TestClient


def test_get_model_classes():
    client = TestClient(ml_api.app)
    r = client.get("/get_model_classes")
    assert r.status_code == 200
    assert "model_classes" in r.json()
    assert "LGBMRegressor" in r.json()["model_classes"]


def test_create_model_with_mocked_s3(monkeypatch):
    # reset globals to keep tests independent
    ml_api.next_model_id = 0
    ml_api.model_hyperparameters = {}

    monkeypatch.setattr(ml_api, "save_model_to_s3", lambda *_args, **_kw: True)

    client = TestClient(ml_api.app)
    r = client.post(
        "/create_model",
        json={"model_type": "LGBMRegressor", "hyperparameters": {"n_estimators": 5}},
    )
    assert r.status_code == 200
    assert r.json()["model_id"] == 0
