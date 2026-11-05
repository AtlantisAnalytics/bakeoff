"""The loopback probe that lets local-server tests skip where loopback is blocked."""

import socket

import fakestore
import pytest


def test_probe_sees_working_loopback(monkeypatch):
    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: Conn())
    assert fakestore._loopback_probe() is True


def test_probe_reports_blocked_loopback(monkeypatch):
    def blocked(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", blocked)
    assert fakestore._loopback_probe() is False


def test_require_loopback_skips_when_blocked(monkeypatch):
    monkeypatch.setattr(fakestore, "loopback_ok", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="loopback"):
        fakestore.require_loopback()
