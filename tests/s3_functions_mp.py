import s3_functions as s3f


def test_save_and_load_model_with_fake_s3(monkeypatch, fake_s3):
    monkeypatch.setattr(s3f, "S3_AVAILABLE", True, raising=False)
    monkeypatch.setattr(s3f, "s3_client", fake_s3, raising=False)
    monkeypatch.setattr(s3f, "BUCKET_NAME", "ml-models", raising=False)

    fake_s3.create_bucket(Bucket="ml-models")

    model_id = 1
    model_obj = {"any": "picklable-object", "v": 123}

    ok = s3f.save_model_to_s3(model_id, model_obj)
    assert ok is True

    loaded = s3f.load_model_from_s3(model_id)
    assert loaded == model_obj
