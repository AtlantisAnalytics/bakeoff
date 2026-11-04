import hashlib
import urllib.error

import pytest

from bakeoff.s3 import S3Client, sign
from bakeoff.store import StoreError, object_key

EMPTY_SHA = hashlib.sha256(b"").hexdigest()
# Expected value copied from the AWS S3 API reference, "Signature Calculations for the
# Authorization Header: Transferring Payload in a Single Chunk", Example: GET Object.
AWS_EXAMPLE_SIGNATURE = "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
SECRET = "s3cr3t-value-for-tests"


@pytest.fixture
def client(fake_store, monkeypatch):
    monkeypatch.setenv("BAKEOFF_STORE_ENDPOINT", fake_store.url)
    monkeypatch.setenv("BAKEOFF_STORE_KEY_ID", "AKIDTEST")
    monkeypatch.setenv("BAKEOFF_STORE_SECRET", SECRET)
    monkeypatch.setenv("BAKEOFF_STORE_BUCKET", fake_store.bucket)
    monkeypatch.setattr("bakeoff.s3.time.sleep", lambda s: None)
    return S3Client.from_env()


def _file(tmp_path, data=b"hello object"):
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_sigv4_known_answer():
    h = sign(
        "GET",
        "https://examplebucket.s3.amazonaws.com/test.txt",
        {"range": "bytes=0-9"},
        EMPTY_SHA,
        key_id="AKIAIOSFODNN7EXAMPLE",
        secret="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        service="s3",
        amz_date="20130524T000000Z",
    )
    assert h["Authorization"].endswith("Signature=" + AWS_EXAMPLE_SIGNATURE)
    assert "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date," in h["Authorization"]
    assert h["x-amz-date"] == "20130524T000000Z"
    assert h["x-amz-content-sha256"] == EMPTY_SHA


def test_put_sends_signed_immutable_object(client, fake_store, tmp_path):
    path, sha = _file(tmp_path)
    key = object_key(sha)
    client.put(key, path, sha)
    assert fake_store.objects[key] == b"hello object"
    method, url_path, headers = fake_store.requests[-1]
    h = {k.lower(): v for k, v in headers.items()}
    assert method == "PUT" and url_path == f"/{fake_store.bucket}/{key}"
    assert h["content-type"] == "application/octet-stream"
    assert h["cache-control"] == "public, max-age=31536000, immutable"
    assert h["x-amz-content-sha256"] == sha
    assert h["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDTEST/")
    assert "/auto/s3/aws4_request" in h["authorization"]
    assert SECRET not in h["authorization"]


def test_put_retries_then_succeeds(client, fake_store, tmp_path):
    path, sha = _file(tmp_path)
    key = object_key(sha)
    fake_store.fail[key] = 503
    fake_store.fail_left[key] = 2
    client.put(key, path, sha)
    assert fake_store.objects[key] == b"hello object"
    assert [r[0] for r in fake_store.requests] == ["PUT"] * 3


def test_put_gives_up_after_three_retries(client, fake_store, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("bakeoff.s3.time.sleep", sleeps.append)
    path, sha = _file(tmp_path)
    key = object_key(sha)
    fake_store.fail[key] = 500
    with pytest.raises(StoreError, match=sha) as e:
        client.put(key, path, sha)
    assert "500" in str(e.value)
    assert len(fake_store.requests) == 4
    assert sleeps == [1, 2, 4]


def test_put_429_retried_but_403_fails_at_once(client, fake_store, tmp_path):
    path, sha = _file(tmp_path)
    key = object_key(sha)
    fake_store.fail[key] = 429
    fake_store.fail_left[key] = 1
    client.put(key, path, sha)
    assert len(fake_store.requests) == 2
    fake_store.requests.clear()
    fake_store.fail[key] = 403
    fake_store.fail_left.pop(key)
    with pytest.raises(StoreError, match="403"):
        client.put(key, path, sha)
    assert len(fake_store.requests) == 1


def test_network_error_retried(client, tmp_path, monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("bakeoff.s3.urllib.request.urlopen", boom)
    path, sha = _file(tmp_path)
    with pytest.raises(StoreError, match=sha):
        client.put(object_key(sha), path, sha)
    assert len(calls) == 4


def test_head_true_false(client, fake_store):
    fake_store.objects["a"] = b"x"
    assert client.head("a") is True
    assert client.head("missing") is False
    assert {k.lower(): v for k, v in fake_store.requests[0][2].items()}[
        "x-amz-content-sha256"
    ] == EMPTY_SHA
    fake_store.fail["denied"] = 403
    with pytest.raises(StoreError, match="403"):
        client.head("denied")


def test_from_env_missing_names_variable(monkeypatch):
    monkeypatch.setenv("BAKEOFF_STORE_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("BAKEOFF_STORE_KEY_ID", "AKIDTEST")
    monkeypatch.delenv("BAKEOFF_STORE_SECRET", raising=False)
    with pytest.raises(StoreError) as e:
        S3Client.from_env()
    assert "BAKEOFF_STORE_SECRET" in str(e.value)
    assert "AKIDTEST" not in str(e.value)


def test_bucket_defaults(monkeypatch):
    monkeypatch.setenv("BAKEOFF_STORE_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("BAKEOFF_STORE_KEY_ID", "k")
    monkeypatch.setenv("BAKEOFF_STORE_SECRET", "s")
    monkeypatch.delenv("BAKEOFF_STORE_BUCKET", raising=False)
    assert S3Client.from_env().bucket == "bakeoff-data"


def test_secret_never_in_repr_or_errors(client, fake_store, tmp_path):
    assert SECRET not in repr(client) and SECRET not in str(client)
    path, sha = _file(tmp_path)
    key = object_key(sha)
    fake_store.fail[key] = 403
    with pytest.raises(StoreError) as e:
        client.put(key, path, sha)
    assert SECRET not in str(e.value)


def test_no_delete_method():
    assert not any(n.lower().startswith("del") for n in dir(S3Client))
