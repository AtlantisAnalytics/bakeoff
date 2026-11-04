import hashlib
import urllib.request

import pytest

from bakeoff import store as store_mod
from bakeoff.ledger import Withdrawal
from bakeoff.store import (
    HashMismatch,
    Store,
    StoreError,
    Unavailable,
    Withdrawn,
    object_key,
    resolve_file,
)


def sha_of(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make(fs, tmp_path, data=b"hello"):
    sha = sha_of(data)
    fs.objects[object_key(sha)] = data
    return sha, Store.from_env(tmp_path)


def gets(fs):
    return [r for r in fs.requests if r[0] == "GET"]


def test_object_key():
    assert object_key("ab" + "0" * 62) == "sha256/ab/ab" + "0" * 62


def test_resolve_downloads_once_then_uses_cache(fake_store, tmp_path):
    sha, st = make(fake_store, tmp_path)
    p = st.resolve(sha)
    assert p.read_bytes() == b"hello"
    assert st.resolve(sha) == p
    assert st.cached(sha) == p
    assert len(gets(fake_store)) == 1


def test_resolve_refuses_wrong_bytes(fake_store, tmp_path):
    sha = sha_of(b"right")
    fake_store.objects[object_key(sha)] = b"wrong"
    st = Store.from_env(tmp_path)
    with pytest.raises(HashMismatch):
        st.resolve(sha)
    assert st.cached(sha) is None
    assert list((st.cache / "sha256").rglob("*")) == [st.cache / "sha256" / sha[:2]]


def test_cached_file_rehashed_on_read(fake_store, tmp_path):
    sha, st = make(fake_store, tmp_path)
    p = st.resolve(sha)
    p.write_bytes(b"corrupt")
    with pytest.raises(HashMismatch):
        st.resolve(sha)


def test_withdrawn_raises_without_network(fake_store, tmp_path):
    sha, st = make(fake_store, tmp_path)
    st.withdrawn[sha] = Withdrawal(sha, "2026-11-01", "leaked key", ("a",))
    with pytest.raises(Withdrawn) as e:
        st.resolve(sha)
    assert e.value.date == "2026-11-01" and e.value.reason == "leaked key"
    assert fake_store.requests == []


def test_404_is_unavailable_with_url(fake_store, tmp_path):
    sha = sha_of(b"missing")
    st = Store.from_env(tmp_path)
    with pytest.raises(Unavailable) as e:
        st.resolve(sha)
    assert fake_store.url in str(e.value) and "404" in str(e.value) and sha in str(e.value)
    assert len(fake_store.requests) == 1


def test_5xx_retried_then_ok(fake_store, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(store_mod.time, "sleep", sleeps.append)
    sha, st = make(fake_store, tmp_path)
    key = object_key(sha)
    fake_store.fail[key] = 503
    fake_store.fail_left[key] = 2
    assert st.resolve(sha).read_bytes() == b"hello"
    assert sleeps == [1, 2]
    assert len(gets(fake_store)) == 3


def test_5xx_gives_up_after_three_retries(fake_store, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(store_mod.time, "sleep", sleeps.append)
    sha, st = make(fake_store, tmp_path)
    fake_store.fail[object_key(sha)] = 500
    with pytest.raises(Unavailable) as e:
        st.resolve(sha)
    assert "500" in str(e.value)
    assert sleeps == [1, 2, 4]


def test_exists(fake_store, tmp_path):
    sha, st = make(fake_store, tmp_path)
    assert st.exists(sha) is True
    assert st.exists(sha_of(b"nope")) is False
    assert {r[0] for r in fake_store.requests} == {"HEAD"}


def test_base_url_with_path_and_slash(tmp_path):
    st = Store("https://x.org/mirror/", tmp_path, {})
    sha = "ab" + "0" * 62
    assert st._url(sha) == f"https://x.org/mirror/sha256/ab/{sha}"


def test_empty_object(fake_store, tmp_path):
    sha, st = make(fake_store, tmp_path, b"")
    p = st.resolve(sha)
    assert p.read_bytes() == b"" and p.stat().st_size == 0


def test_interrupted_download_leaves_no_cache_file(fake_store, tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.time, "sleep", lambda s: None)
    sha, st = make(fake_store, tmp_path, b"x" * 100)
    real = urllib.request.urlopen

    class Broken:
        def __init__(self, resp):
            self.resp, self.n = resp, 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.resp.close()

        def read(self, n):
            self.n += 1
            if self.n > 1:
                raise ConnectionResetError("cut")
            return self.resp.read(10)

    monkeypatch.setattr(store_mod.urllib.request, "urlopen", lambda *a, **k: Broken(real(*a, **k)))
    with pytest.raises(Unavailable):
        st.resolve(sha)
    assert [p for p in (st.cache / "sha256").rglob("*") if p.is_file()] == []


def test_unwritable_cache_names_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    st = Store("https://x.org", blocker / "cache", {})
    with pytest.raises(StoreError) as e:
        st.resolve("ab" + "0" * 62)
    assert str(blocker / "cache") in str(e.value)


def test_resolve_file_prefers_local(tmp_path, fake_store):
    sha, st = make(fake_store, tmp_path)
    (tmp_path / "f.txt").write_text("local")
    assert resolve_file(tmp_path, "f.txt", sha, st) == tmp_path / "f.txt"
    assert fake_store.requests == []
    assert resolve_file(tmp_path, "g.txt", sha, st).read_bytes() == b"hello"
    with pytest.raises(Unavailable):
        resolve_file(tmp_path, "g.txt", sha, None)


def test_fake_store_put_and_head_paths(fake_store):
    key = "sha256/aa/" + "a" * 64
    req = urllib.request.Request(
        f"{fake_store.url}/{fake_store.bucket}/{key}", data=b"body", method="PUT"
    )
    urllib.request.urlopen(req).close()
    assert fake_store.objects[key] == b"body"
    for path in (key, f"{fake_store.bucket}/{key}"):
        urllib.request.urlopen(
            urllib.request.Request(f"{fake_store.url}/{path}", method="HEAD")
        ).close()
    assert [r[0] for r in fake_store.requests] == ["PUT", "HEAD", "HEAD"]
