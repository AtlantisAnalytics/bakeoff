import hashlib
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from weekfix import KEPT_RUN, make_bundle, push_all, split

from bakeoff import cli
from bakeoff.archive import ArchiveError, build_archive, load_archive_rows, zenodo_publish
from bakeoff.store import HashMismatch, Store, StoreError, object_key

LAYOUT = (
    '[[layout]]\nfrom = "2026-11-02"\nkeep = "all"\n[[layout]]\nfrom = "2026-11-30"\n'
    'run = ["bundle.json", "speed.jsonl", "fit.jsonl"]\njudge = ["bundle.json"]\n'
)
WEEKS = ("2026-11-30", "2026-12-07", "2026-12-14")


def _root(tmp_path, fs, weeks=WEEKS):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "results" / "layout.toml").write_text(LAYOUT)
    bundles = []
    for i, week in enumerate(weeks):
        b = make_bundle(
            root / "results" / week / "runs",
            label=f"M{i}",
            weights="w",
            envelope="e",
            grades={"general/x": [True, i % 2 == 0, False]},
        )
        push_all(fs, b)
        split(b, KEPT_RUN | {"bundle.json"})
        bundles.append(b)
    return root, bundles


def _files(b):
    return json.loads((b / "bundle.json").read_text())["files"]


def _stored(b):
    return {rel: sha for rel, sha in _files(b).items() if rel not in KEPT_RUN}


def _tars(out):
    return sorted(out.glob("*.tar"))


def test_archive_excludes_withdrawn(tmp_path, fake_store):
    root, bundles = _root(tmp_path, fake_store)
    rel, sha = next(iter(_stored(bundles[0]).items()))
    bundle_path = bundles[0].relative_to(root).as_posix()
    (root / "results" / "withdrawn.toml").write_text(
        f'[[withdrawn]]\nsha256 = "{sha}"\ndate = "2026-12-20"\nreason = "test"\n'
        f'bundles = ["{bundle_path}"]\n'
    )
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    manifest = (out / "MANIFEST.sha256").read_text()
    assert sha not in manifest
    for tar in _tars(out):
        with tarfile.open(tar) as t:
            assert object_key(sha) not in t.getnames()
    other = next(s for r, s in _stored(bundles[1]).items() if s != sha)
    assert other in manifest


def test_one_tar_per_month_and_under_100_files(tmp_path, fake_store):
    root, bundles = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    assert [p.name for p in _tars(out)] == ["2026-11.tar", "2026-12.tar"]
    assert len(list(out.iterdir())) == 4 <= 100
    lines = (out / "MANIFEST.sha256").read_text().splitlines()
    assert lines == sorted(lines)
    sha, size, paths = lines[0].split("  ")
    assert len(sha) == 64 and size.isdigit() and paths.startswith("results/")
    with tarfile.open(out / "2026-11.tar") as t:
        names = t.getnames()
        assert names == sorted(names)
        for m in t.getmembers():
            assert (m.mtime, m.uid, m.gid, m.uname, m.gname) == (0, 0, 0, "", "")
    # each object once, in all
    all_names = []
    for tar in _tars(out):
        with tarfile.open(tar) as t:
            all_names += t.getnames()
    assert len(all_names) == len(set(all_names)) == len(lines)


def test_rebuild_is_byte_identical(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    a = build_archive(root, 2026, tmp_path / "a", Store.from_env(root))
    b = build_archive(root, 2026, tmp_path / "b", Store.from_env(root))
    for f in sorted(p.name for p in a.iterdir()):
        assert (a / f).read_bytes() == (b / f).read_bytes()


def test_year_without_split_weeks_is_an_error(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    with pytest.raises(ArchiveError, match="2025"):
        build_archive(root, 2025, tmp_path / "out", Store.from_env(root))
    assert not (tmp_path / "out").exists()


def test_manifest_round_trip_into_cache(tmp_path, fake_store, monkeypatch):
    root, bundles = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    fresh = Store(fake_store.url, tmp_path / "fresh-cache", {})
    shas = {s for b in bundles for s in _files(b).values()}
    assert all(fresh.cached(s) is None for s in shas)
    assert fresh.load_archive(out) == len(shas)
    assert all(fresh.cached(s) for s in shas)
    before = len(fake_store.requests)
    for s in shas:
        fresh.resolve(s)
    assert len(fake_store.requests) == before


def _rewrite_member(tar_path, name, data):
    with tarfile.open(tar_path) as t:
        members = [(m, t.extractfile(m).read()) for m in t.getmembers()]
    with tarfile.open(tar_path, "w") as t:
        for m, body in members:
            if m.name == name:
                body = data
                m.size = len(body)
            t.addfile(m, io.BytesIO(body))


def test_tampered_tar_member_refused(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    tar = out / "2026-11.tar"
    with tarfile.open(tar) as t:
        name = t.getnames()[0]
    _rewrite_member(tar, name, b"evil")
    fresh = Store(fake_store.url, tmp_path / "fresh-cache", {})
    with pytest.raises(HashMismatch):
        fresh.load_archive(out)
    assert fresh.cached(name.rsplit("/", 1)[1]) is None


def test_same_size_tamper_refused(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    tar = out / "2026-11.tar"
    with tarfile.open(tar) as t:
        m = t.getmembers()[0]
        body = t.extractfile(m).read()
    _rewrite_member(tar, m.name, bytes(b ^ 1 for b in body))
    fresh = Store(fake_store.url, tmp_path / "fresh-cache", {})
    with pytest.raises(HashMismatch):
        fresh.load_archive(out)
    assert fresh.cached(m.name.rsplit("/", 1)[1]) is None


def test_build_refuses_a_used_directory(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    out = tmp_path / "out"
    out.mkdir()
    (out / "2025-03.tar").write_bytes(b"stale")
    with pytest.raises(ArchiveError, match="not empty"):
        build_archive(root, 2026, out, Store.from_env(root))
    assert (out / "2025-03.tar").read_bytes() == b"stale"


def test_publish_refuses_stray_or_stale_files(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    with FakeZenodo() as z:
        (out / "2025-03.tar").write_bytes(b"stale")
        with pytest.raises(ArchiveError, match="unexpected"):
            zenodo_publish(out, 2026, None, z.url, "tok")
        (out / "2025-03.tar").unlink()
        (out / "2026-01.tar").write_bytes(b"stale")
        with pytest.raises(ArchiveError):
            zenodo_publish(out, 2026, None, z.url, "tok")
        (out / "2026-01.tar").unlink()
        (out / "notes.txt").write_text("x")
        with pytest.raises(ArchiveError, match="unexpected"):
            zenodo_publish(out, 2026, None, z.url, "tok")
        assert z.calls == []


def test_member_not_in_manifest_refused(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    body = b"extra"
    sha = hashlib.sha256(body).hexdigest()
    with tarfile.open(out / "2026-11.tar", "a") as t:
        info = tarfile.TarInfo(object_key(sha))
        info.size = len(body)
        t.addfile(info, io.BytesIO(body))
    fresh = Store(fake_store.url, tmp_path / "fresh-cache", {})
    with pytest.raises(StoreError, match="manifest"):
        fresh.load_archive(out)


class FakeZenodo:
    def __init__(self):
        self.calls: list[tuple[str, str, dict, bytes]] = []
        self.published_meta = None
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _do(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                owner.calls.append((self.command, self.path, dict(self.headers), body))
                code, resp = owner.answer(self.command, self.path, body)
                raw = json.dumps(resp).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_POST = do_PUT = do_GET = _do

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), H)  # scrub-ok
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"  # scrub-ok

    def answer(self, method, path, body):
        if (method, path) == ("POST", "/api/deposit/depositions"):
            return 201, {
                "id": 77,
                "links": {"bucket": f"{self.url}/api/files/bkt"},
            }
        if method == "PUT" and path.startswith("/api/files/bkt/"):
            return 201, {"key": path.rsplit("/", 1)[1]}
        if (method, path) == ("PUT", "/api/deposit/depositions/77"):
            self.published_meta = json.loads(body)["metadata"]
            return 200, {"id": 77}
        if (method, path) == ("POST", "/api/deposit/depositions/77/actions/publish"):
            return 202, {"id": 77, "doi": "10.5281/zenodo.4242"}
        return 404, {}

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def test_zenodo_upload_against_fake(tmp_path, fake_store, monkeypatch):
    root, _ = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    with FakeZenodo() as z:
        doi = zenodo_publish(out, 2026, "10.5281/zenodo.1111", z.url, "s3cret-token")
    assert doi == "10.5281/zenodo.4242"
    seq = [(m, p) for m, p, _, _ in z.calls]
    names = sorted(p.name for p in out.iterdir())
    assert seq == [
        ("POST", "/api/deposit/depositions"),
        *[("PUT", f"/api/files/bkt/{n}") for n in names],
        ("PUT", "/api/deposit/depositions/77"),
        ("POST", "/api/deposit/depositions/77/actions/publish"),
    ]
    assert all(h["Authorization"] == "Bearer s3cret-token" for _, _, h, _ in z.calls)
    put = {p.rsplit("/", 1)[1]: b for m, p, _, b in z.calls if p.startswith("/api/files")}
    assert put["MANIFEST.sha256"] == (out / "MANIFEST.sha256").read_bytes()
    meta = z.published_meta
    assert meta["title"] == "bakeoff.si raw data, 2026"
    assert meta["license"] == "cc-by-4.0"
    assert meta["upload_type"] == "dataset" and meta["access_right"] == "open"
    assert meta["communities"] == [{"identifier": "bakeoff-si"}]
    assert meta["related_identifiers"] == [
        {"identifier": "10.5281/zenodo.1111", "relation": "continues"}
    ]
    assert meta["creators"]


def test_zenodo_error_never_prints_token(tmp_path, fake_store):
    root, _ = _root(tmp_path, fake_store)
    out = build_archive(root, 2026, tmp_path / "out", Store.from_env(root))
    with FakeZenodo() as z:
        z.answer = lambda *a: (401, {"message": "no"})
        with pytest.raises(ArchiveError) as e:
            zenodo_publish(out, 2026, None, z.url, "s3cret-token")
    assert "s3cret-token" not in str(e.value)


def test_archive_rows(tmp_path):
    root = tmp_path
    (root / "results").mkdir()
    (root / "results" / "archive.toml").write_text(
        '[[archive]]\nyear = 2026\ndoi = "10.5281/zenodo.1"\nmanifest_sha256 = "' + "a" * 64 + '"\n'
    )
    assert load_archive_rows(root)[2026].doi == "10.5281/zenodo.1"
    (root / "results" / "archive.toml").write_text('[[archive]]\nyear = 2026\ndoi = ""\n')
    with pytest.raises(ArchiveError):
        load_archive_rows(root)


def test_cli_build_and_fetch_archive_dir(tmp_path, fake_store, monkeypatch, capsys):
    root, bundles = _root(tmp_path, fake_store)
    out = tmp_path / "out"
    assert cli.main(["archive", "build", "2026", "--out", str(out), "--root", str(root)]) == 0
    manifest_sha = hashlib.sha256((out / "MANIFEST.sha256").read_bytes()).hexdigest()
    fake_store.objects.clear()
    fresh = tmp_path / "fresh"
    monkeypatch.setenv("BAKEOFF_CACHE", str(fresh))
    bundle = str(bundles[0])
    argv = ["store", "fetch", bundle, "--root", str(root), "--archive", "2026"]
    assert cli.main([*argv, "--archive-dir", str(out)]) == 0
    assert "fetched 0" in capsys.readouterr().out
    # a recorded manifest sha that does not match refuses the archive
    (root / "results" / "archive.toml").write_text(
        f'[[archive]]\nyear = 2026\ndoi = "10.5281/zenodo.1"\nmanifest_sha256 = "{"0" * 64}"\n'
    )
    assert cli.main([*argv, "--archive-dir", str(out)]) == 1
    (root / "results" / "archive.toml").write_text(
        f'[[archive]]\nyear = 2026\ndoi = "10.5281/zenodo.1"\nmanifest_sha256 = "{manifest_sha}"\n'
    )
    assert cli.main([*argv, "--archive-dir", str(out)]) == 0
