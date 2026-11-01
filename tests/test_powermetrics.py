import plistlib
import time
from pathlib import Path

import pytest

from bakeoff.powermetrics import POWERMETRICS_ARGV, PowerStream, parse_plists

FIXTURE = Path(__file__).parent / "data" / "env" / "powermetrics.plist"


def plist_bytes(combined_mw: float) -> bytes:
    return plistlib.dumps({"processor": {"combined_power": combined_mw}})


def test_argv_is_exact():
    assert POWERMETRICS_ARGV == [
        "sudo",
        "-n",
        "/usr/bin/powermetrics",
        "--samplers",
        "cpu_power,gpu_power",
        "-i",
        "500",
        "-f",
        "plist",
    ]


def test_parse_plists_fixture():
    watts, tail = parse_plists(FIXTURE.read_bytes())
    assert len(watts) >= 10
    assert all(0.05 <= w <= 200 for w in watts)
    assert watts[0] == pytest.approx(0.528631)


def test_parse_plists_sums_components_without_combined():
    doc = plistlib.dumps({"processor": {"cpu_power": 100.0, "gpu_power": 50.0, "ane_power": 10.0}})
    assert parse_plists(doc + b"\0") == ([pytest.approx(0.16)], b"")


def test_parse_plists_keeps_partial_tail():
    a, b = plist_bytes(1000.0), plist_bytes(2000.0)
    buf = a + b"\0" + b[:20]
    watts, tail = parse_plists(buf)
    assert watts == [pytest.approx(1.0)]
    assert tail == b[:20]
    watts, tail = parse_plists(tail + b[20:] + b"\0")
    assert watts == [pytest.approx(2.0)]
    assert tail == b""


class FakeProc:
    def __init__(self, chunks=(), returncode=None):
        self.stdout = _Pipe(list(chunks))
        self.returncode = returncode
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


class _Pipe:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def close(self):
        self.closed = True

    def read1(self, n):
        return self._chunks.pop(0) if self._chunks else b""


def test_powerstream_not_permitted():
    ps = PowerStream(popen=lambda *a, **k: FakeProc(returncode=1))
    assert ps.start() is False
    assert ps.latest() is None
    ps.stop()


def test_powerstream_reads_latest(monkeypatch):
    monkeypatch.setattr("bakeoff.powermetrics.START_GRACE_S", 0.1)
    proc = FakeProc([plist_bytes(1500.0) + b"\0"])
    ps = PowerStream(popen=lambda *a, **k: proc)
    assert ps.start() is True
    for _ in range(100):
        if ps.latest() is not None:
            break
        time.sleep(0.01)
    assert ps.latest() == pytest.approx(1.5)
    ps.stop()
    assert proc.terminated


def test_power_stale_reading_not_reused():
    now = [100.0]
    ps = PowerStream(popen=lambda *a, **k: FakeProc(), clock=lambda: now[0])
    ps._set(2.5)
    assert ps.latest() == 2.5
    now[0] = 103.0
    assert ps.latest() is None


def test_stop_reaps_and_closes_even_when_already_exited():
    proc = FakeProc(returncode=1)
    ps = PowerStream(popen=lambda *a, **k: proc)
    assert ps.start() is False
    ps.stop()
    assert proc.waited and proc.stdout.closed


def test_stop_closes_stdout_of_running_process(monkeypatch):
    monkeypatch.setattr("bakeoff.powermetrics.START_GRACE_S", 0.1)
    proc = FakeProc()
    ps = PowerStream(popen=lambda *a, **k: proc)
    assert ps.start() is True
    ps.stop()
    assert proc.terminated and proc.waited and proc.stdout.closed
