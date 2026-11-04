"""A loopback stand-in for the object store: public GET/HEAD and S3 path-style GET/HEAD/PUT."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

BUCKET = "bkt"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _key(self) -> str:
        path = self.path.split("?")[0].lstrip("/")
        prefix = self.server.bucket + "/"
        return path[len(prefix) :] if path.startswith(prefix) else path

    def _record(self) -> str:
        self.server.owner.requests.append((self.command, self.path, dict(self.headers)))
        return self._key()

    def _forced(self, key: str) -> int | None:
        owner = self.server.owner
        status = owner.fail.get(key)
        if status is None:
            return None
        left = owner.fail_left.get(key)
        if left is not None:
            if left <= 0:
                return None
            owner.fail_left[key] = left - 1
        return status

    def _serve(self, body_too: bool) -> None:
        key = self._record()
        forced = self._forced(key)
        if forced is not None:
            self.send_response(forced)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = self.server.owner.objects.get(key)
        if data is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body_too:
            self.wfile.write(data)

    def do_GET(self):
        self._serve(True)

    def do_HEAD(self):
        self._serve(False)

    def do_PUT(self):
        key = self._record()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        forced = self._forced(key)
        if forced is not None:
            self.send_response(forced)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.server.owner.objects[key] = body
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class FakeStore:
    bucket = BUCKET

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.requests: list[tuple[str, str, dict]] = []
        self.fail: dict[str, int] = {}
        self.fail_left: dict[str, int] = {}  # optional: serve the forced status this many times
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)  # scrub-ok
        self._server.owner = self
        self._server.bucket = self.bucket
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"  # scrub-ok

    def __enter__(self) -> "FakeStore":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    with FakeStore() as fs:
        monkeypatch.setenv("BAKEOFF_STORE_URL", fs.url)
        monkeypatch.setenv("BAKEOFF_CACHE", str(tmp_path / "cache"))
        yield fs
