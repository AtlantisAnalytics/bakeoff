"""An in-process stand-in for the object store: public GET/HEAD and S3 path-style GET/HEAD/PUT.

Test servers are reached in process: a urllib connection to a routed loopback address gets one
end of a socketpair, and the server's handler runs on the other end in a thread. HTTP is parsed
for real, but no IP socket is opened, so these tests also run where loopback is blocked (Athena's
publish runner). Any address without a route goes out over the network as usual.
"""

import http.client
import itertools
import socket
import threading
import types
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler

import pytest

BUCKET = "bkt"

_ROUTES: dict[str, Callable[[socket.socket], None]] = {}  # "host:port" -> serves one connection
_PORTS = itertools.count(50000)


class _Connection(http.client.HTTPConnection):
    def connect(self):
        serve = _ROUTES.get(f"{self.host}:{self.port}")
        if serve is None:
            return super().connect()
        self.sock, theirs = socket.socketpair()
        if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            self.sock.settimeout(self.timeout)
        threading.Thread(target=_serve_once, args=(serve, theirs), daemon=True).start()


def _serve_once(serve: Callable[[socket.socket], None], sock: socket.socket) -> None:
    with sock:
        serve(sock)


class _RoutingHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_Connection, req)


urllib.request.install_opener(urllib.request.build_opener(_RoutingHandler))


class Route:
    """An in-process address: each connection to `url` is served by `serve(sock)`, which reads
    the request from sock and writes the reply; the connection closes when it returns."""

    def __init__(self, serve: Callable[[socket.socket], None]):
        self._addr = f"127.0.0.1:{next(_PORTS)}"  # scrub-ok
        self.url = f"http://{self._addr}"
        _ROUTES[self._addr] = serve

    def close(self) -> None:
        _ROUTES.pop(self._addr, None)


def http_route(handler: type[BaseHTTPRequestHandler], **server_attrs) -> Route:
    """A Route served by an http.server handler class; `self.server` carries server_attrs."""
    server = types.SimpleNamespace(**server_attrs)
    return Route(lambda sock: handler(sock, ("127.0.0.1", 0), server))  # scrub-ok


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
        self._route = http_route(_Handler, owner=self, bucket=self.bucket)
        self.url = self._route.url

    def __enter__(self) -> "FakeStore":
        return self

    def __exit__(self, *exc) -> None:
        self._route.close()


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    with FakeStore() as fs:
        monkeypatch.setenv("BAKEOFF_STORE_URL", fs.url)
        monkeypatch.setenv("BAKEOFF_CACHE", str(tmp_path / "cache"))
        yield fs
