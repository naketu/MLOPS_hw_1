import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import ml_api


@pytest.fixture()
def client_with_inmemory_s3(monkeypatch):
    
    ml_api.next_model_id = 0
    ml_api.models = {}
    ml_api.model_hyperparameters = {}

    storage = {}

    def _save(mid, model):
        storage[int(mid)] = model
        return True

    def _load(mid):
        return storage.get(int(mid), False)

    def _delete(mid):
        mid = int(mid)
        existed = mid in storage
        storage.pop(mid, None)
        return existed

    def _list_ids():
        return [str(k) for k in sorted(storage.keys())]

    monkeypatch.setattr(ml_api, "save_model_to_s3", _save, raising=False)
    monkeypatch.setattr(ml_api, "load_model_from_s3", _load, raising=False)
    monkeypatch.setattr(ml_api, "delete_model_from_s3", _delete, raising=False)
    monkeypatch.setattr(ml_api, "get_model_ids_from_s3", _list_ids, raising=False)

    return TestClient(ml_api.app)


def _toy_classification_data():
    X = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 0.2, 1.2, 0.1, 1.1, 0.3, 1.3],
            "f2": [1.0, 0.0, 1.1, 0.1, 1.2, 0.2, 1.3, 0.3],
        }
    )
    y = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1, 0, 1]})
    return X, y


def test_train_new_model_with_hyperparams(client_with_inmemory_s3):
    client = client_with_inmemory_s3
    X_train, Y_classification = _toy_classification_data()

    hyperparameters_classifier = {
        "n_estimators": 10,
        "learning_rate": 0.2,
        "min_child_samples": 1,
        "num_leaves": 7,
        "verbosity": -1,
    }

    data = {
        "model_type": "LGBMClassifier",
        "hyperparameters": hyperparameters_classifier,
        "X_data": X_train.to_json(orient="records"),
        "Y_data": Y_classification.to_json(orient="records"),
    }

    r = client.post("/train_new_model", json=data)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "trained"
    assert "model_id" in body


def test_predict_train_retrain_delete_healthcheck_flow(client_with_inmemory_s3):
    client = client_with_inmemory_s3
    X_train, Y_classification = _toy_classification_data()

    # 1) train_new_model (create+train) -> get model_id
    r1 = client.post(
        "/train_new_model",
        json={
            "model_type": "LGBMClassifier",
            "hyperparameters": {"n_estimators": 10, "verbosity": -1},
            "X_data": X_train.to_json(orient="records"),
            "Y_data": Y_classification.to_json(orient="records"),
        },
    )
    assert r1.status_code == 200
    model_id = r1.json()["model_id"]

    # 2) predict
    r2 = client.post(
        "/predict",
        json={
            "model_id": model_id,
            "X_data": X_train.to_json(orient="records"),
        },
    )
    assert r2.status_code == 200
    preds = r2.json()["prediction"]
    assert isinstance(preds, list)
    assert len(preds) == len(X_train)

    # 3) retrain existing model by id (train_model)
    r3 = client.post(
        "/train_model",
        json={
            "model_id": model_id,
            "X_data": X_train[:20].to_json(orient="records"),
            "Y_data": Y_classification[:20].to_json(orient="records"),
        },
    )
    assert r3.status_code == 200
    assert r3.json()["status"] == "trained"

    # 4) delete model
    r4 = client.delete(f"/delete_model/{model_id}")
    assert r4.status_code == 200
    assert r4.json()["status"] in ("success", "failed")  # depends on storage

    # 5) healthcheck
    r5 = client.get("/healthcheck")
    assert r5.status_code == 200
    assert r5.json()["status"] == "ok"
