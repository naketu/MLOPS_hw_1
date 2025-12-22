from __future__ import annotations

import io
import types
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class FakeS3:
    class _Exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self._storage: dict[tuple[str, str], bytes] = {}
        self._buckets: set[str] = set()
        self.exceptions = FakeS3._Exceptions()

    def create_bucket(self, Bucket: str):
        self._buckets.add(Bucket)
        return {}

    def put_object(self, Bucket: str, Key: str, Body: bytes):
        if Bucket not in self._buckets:
            self._buckets.add(Bucket)
        self._storage[(Bucket, Key)] = Body
        return {}

    def get_object(self, Bucket: str, Key: str):
        try:
            data = self._storage[(Bucket, Key)]
        except KeyError as e:
            raise self.exceptions.NoSuchKey(str(e))
        return {"Body": io.BytesIO(data)}

    def delete_object(self, Bucket: str, Key: str):
        self._storage.pop((Bucket, Key), None)
        return {}

    def list_objects_v2(self, Bucket: str, Prefix: str):
        keys = []
        for (b, k), _ in self._storage.items():
            if b == Bucket and k.startswith(Prefix):
                keys.append({"Key": k})
        return {"Contents": keys} if keys else {}


@pytest.fixture()
def fake_s3():
    return FakeS3()
