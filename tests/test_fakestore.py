"""The in-process routes that let the local-server tests run where loopback is blocked."""

import socket
import urllib.error
import urllib.request

import fakestore
import pytest


@pytest.fixture
def no_network(monkeypatch):
    """Every IP connection fails, as in Athena's publish runner (IPAddressDeny covers loopback)."""
    tried = []

    def blocked(address, *a, **k):
        tried.append(address)
        raise TimeoutError("timed out (network blocked)")

    monkeypatch.setattr(socket, "create_connection", blocked)
    return tried


def test_fake_store_needs_no_network(no_network):
    with fakestore.FakeStore() as fs:
        fs.objects["a/b"] = b"payload"
        url = f"{fs.url}/{fs.bucket}/a/b"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.read() == b"payload"
        req = urllib.request.Request(f"{fs.url}/{fs.bucket}/c", data=b"put", method="PUT")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
        assert fs.objects["c"] == b"put"
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{fs.url}/{fs.bucket}/missing", timeout=5)
        assert err.value.code == 404
    assert no_network == []


def test_raw_route_serves_bytes_then_closes(no_network):
    reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"

    def serve(sock):
        sock.recv(65536)
        sock.sendall(reply)

    route = fakestore.Route(serve)
    try:
        with urllib.request.urlopen(route.url, timeout=5) as resp:
            assert resp.read() == b"ok"
    finally:
        route.close()
    assert no_network == []


def test_unrouted_addresses_use_the_network(no_network):
    with fakestore.FakeStore() as fs:
        url = fs.url
    with pytest.raises(urllib.error.URLError):  # the route is gone with the store
        urllib.request.urlopen(url, timeout=5)
    assert len(no_network) == 1
