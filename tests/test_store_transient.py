import hashlib
import json
import socket
import threading

import pytest
from fakestore import require_loopback
from weekfix import KEPT_RUN, make_bundle, push_all, split

from bakeoff import cli
from bakeoff import store as store_mod
from bakeoff.bundle import verify_bundle
from bakeoff.store import HashMismatch, Store, StoreError, Unavailable, object_key

BODY = b"x" * 100
SHA = hashlib.sha256(BODY).hexdigest()
LAYOUT = (
    '[[layout]]\nfrom = "2026-11-02"\nkeep = "all"\n[[layout]]\nfrom = "2026-11-30"\n'
    'run = ["bundle.json", "speed.jsonl", "fit.jsonl"]\njudge = ["bundle.json"]\n'
)


class RawServer:
    """Answers every request with fixed bytes, then closes the connection."""

    def __init__(self, reply: bytes):
        require_loopback()
        self.reply, self.hits = reply, 0
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))  # scrub-ok
        self.sock.listen(8)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"  # scrub-ok
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            conn.recv(65536)
            self.hits += 1
            conn.sendall(self.reply)
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.sock.close()


SHORT = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n" + BODY[:50]
CHUNKED = (
    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n64\r\n" + BODY[:50]
)
GARBAGE = b"NOT HTTP AT ALL\r\n\r\n"


@pytest.fixture
def sleeps(monkeypatch):
    out = []
    monkeypatch.setattr(store_mod.time, "sleep", out.append)
    return out


@pytest.mark.parametrize("reply", [SHORT, CHUNKED])
def test_truncated_body_is_retried_then_unavailable(reply, tmp_path, sleeps):
    with RawServer(reply) as srv:
        st = Store(srv.url, tmp_path / "c", {})
        with pytest.raises(Unavailable) as e:
            st.resolve(SHA)
        assert srv.hits == 4
    assert not isinstance(e.value, HashMismatch)
    assert sleeps == [1, 2, 4]
    assert st.cached(SHA) is None
    assert list((st.cache / "sha256").rglob("*")) == [st.cache / "sha256" / SHA[:2]]


def test_full_length_wrong_body_is_still_hash_mismatch(tmp_path, sleeps):
    reply = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n" + b"y" * 100
    with RawServer(reply) as srv:
        with pytest.raises(HashMismatch):
            Store(srv.url, tmp_path / "c", {}).resolve(SHA)
        assert srv.hits == 1


def test_bad_status_line_is_transient_for_exists(tmp_path, sleeps):
    with RawServer(GARBAGE) as srv:
        with pytest.raises(Unavailable):
            Store(srv.url, tmp_path / "c", {}).exists(SHA)
        assert srv.hits == 4
    assert sleeps == [1, 2, 4]


def test_verify_bundle_reports_truncated_download(tmp_path, sleeps):
    with RawServer(SHORT) as srv:
        d = tmp_path / "b"
        d.mkdir()
        (d / "bundle.json").write_text(
            json.dumps({"files": {"quality/a.jsonl": SHA}, "status": "complete"})
        )
        st = Store(srv.url, tmp_path / "c", {})
        found = verify_bundle(d, None, None, kept=frozenset({"bundle.json"}), store=st)
    assert "unavailable" in [f.kind for f in found]
    assert "hash_mismatch" not in [f.kind for f in found]


BAD_SHAS = [12345, "A" * 64, "ab", "../../../escaped/x", None, ["a"]]


@pytest.mark.parametrize("bad", BAD_SHAS, ids=repr)
def test_bad_sha_in_bundle_is_a_finding_and_never_resolved(bad, tmp_path):
    with RawServer(GARBAGE) as srv:
        d = tmp_path / "b"
        d.mkdir()
        (d / "bundle.json").write_text(
            json.dumps({"files": {"quality/a.jsonl": bad}, "status": "complete"})
        )
        st = Store(srv.url, tmp_path / "c", {})
        found = verify_bundle(d, None, None, kept=frozenset({"bundle.json"}), store=st)
        assert srv.hits == 0
    assert any(f.kind == "unreadable" and "bad sha256" in f.detail for f in found)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b"]


@pytest.mark.parametrize("bad", BAD_SHAS[:4], ids=repr)
def test_store_refuses_bad_sha(bad, tmp_path):
    with RawServer(GARBAGE) as srv:
        st = Store(srv.url, tmp_path / "c", {})
        for call in (st.resolve, st.exists, st.cached, store_mod.object_key):
            with pytest.raises(StoreError):
                call(bad)
        assert srv.hits == 0
    assert not (tmp_path / "c").exists() and sorted(p.name for p in tmp_path.iterdir()) == []


def test_materialize_and_cli_listing_refuse_bad_sha(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    (d / "bundle.json").write_text(json.dumps({"files": {"quality/a.jsonl": "../x"}}))
    st = Store("http://unused.invalid", tmp_path / "c", {})
    with pytest.raises(StoreError):
        store_mod.materialize(d, tmp_path / "out", st)
    with pytest.raises(StoreError):
        cli._listed(d)


def test_ledger_listing_ignores_bad_sha_values(tmp_path):
    from bakeoff.ledgercheck import _listed

    d = tmp_path / "b"
    d.mkdir()
    (d / "bundle.json").write_text(json.dumps({"files": {"quality/a.jsonl": 5}}))
    assert _listed(d) == {}


def _audit_root(tmp_path, fake_store):
    root = tmp_path / "repo"
    run = make_bundle(
        root / "results" / "2026-11-30" / "runs",
        label="A",
        weights="w",
        envelope="e",
        grades={"general/x": [True, False]},
    )
    (root / "results" / "layout.toml").write_text(LAYOUT)
    push_all(fake_store, run)
    split(run, KEPT_RUN | {"bundle.json"})
    return root, run


def test_audit_survives_unavailable_objects(fake_store, tmp_path, capsys, sleeps):
    root, run = _audit_root(tmp_path, fake_store)
    second = make_bundle(
        root / "results" / "2026-11-30" / "judge",
        label="B",
        weights="w2",
        envelope="e",
        grades={"general/x": [True]},
    )
    push_all(fake_store, second)
    split(second, {"bundle.json"})
    files = json.loads((run / "bundle.json").read_text())["files"]
    stored = [s for r, s in files.items() if r not in KEPT_RUN | {"bundle.json"}]
    fake_store.fail[object_key(stored[0])] = 403
    fake_store.requests.clear()
    assert cli.main(["store", "audit", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert f"unavailable {stored[0]}" in out and "audited 2 bundles" in out
    heads = [p for m, p, _ in fake_store.requests if m == "HEAD"]
    assert len(set(heads)) > 1


def test_rehash_reports_unavailable_and_continues(fake_store, tmp_path, capsys, monkeypatch):
    root, run = _audit_root(tmp_path, fake_store)
    real = Store._download
    seen = []

    def flaky(self, sha, path):
        seen.append(sha)
        if len(seen) == 1:
            raise Unavailable(sha, "u", 403)
        return real(self, sha, path)

    monkeypatch.setattr(Store, "_download", flaky)
    assert cli.main(["store", "audit", "--root", str(root), "--rehash", "99"]) == 1
    out = capsys.readouterr().out
    assert f"unavailable {seen[0]} " in out and len(seen) > 1


def _zenodo_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "zenodo_upload.py"
    spec = importlib.util.spec_from_file_location("zenodo_upload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_zenodo_upload_refuses_before_march_of_next_year(tmp_path, capsys, monkeypatch):
    from datetime import date

    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    main = _zenodo_script().main
    argv = [str(tmp_path), "--year", "2026"]
    for early in (date(2026, 12, 31), date(2027, 2, 28)):
        assert main(argv, today=early) == 1
        assert "before 1 March 2027" in capsys.readouterr().err
    assert main(argv, today=date(2027, 3, 1)) == 1
    assert "ZENODO_TOKEN" in capsys.readouterr().err
