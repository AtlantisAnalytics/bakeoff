"""Measured power on macOS: stream powermetrics plists and keep the latest reading.

The sudoers line on the lab Mac permits exactly the command in POWERMETRICS_ARGV (everything after
the sudo options), so it must match byte for byte. `-k` makes sudo ignore a cached ticket, so on a
contributor's Mac without that sudoers line it never silently starts a root powermetrics; `-n` means
it never prompts. powermetrics writes one plist per interval, separated by NUL bytes. Nothing here
raises into the run: if the stream is not permitted or dies, power is simply None.
"""

import math
import plistlib
import subprocess
import threading
import time
from collections.abc import Callable

POWERMETRICS_ARGV = [
    "sudo",
    "-k",
    "-n",
    "/usr/bin/powermetrics",
    "--samplers",
    "cpu_power,gpu_power",
    "-i",
    "500",
    "-f",
    "plist",
]
START_GRACE_S = 2.0
STOP_TIMEOUT_S = 2.0
CHUNK_BYTES = 64 * 1024


def _watts(doc: object) -> float | None:
    proc = doc.get("processor") if isinstance(doc, dict) else None
    if not isinstance(proc, dict):
        return None
    try:
        if proc.get("combined_power") is not None:
            mw = float(proc["combined_power"])
        else:
            mw = sum(float(proc[k]) for k in ("cpu_power", "gpu_power", "ane_power"))
    except (KeyError, TypeError, ValueError):
        return None
    return mw / 1000 if math.isfinite(mw) else None


def parse_plists(buf: bytes) -> tuple[list[float], bytes]:
    """Watts for each complete (NUL-terminated) plist in buf, and the unconsumed tail."""
    *complete, tail = buf.split(b"\0")
    out = []
    for chunk in complete:
        if not chunk.strip():
            continue
        try:
            watts = _watts(plistlib.loads(chunk))
        except Exception:  # noqa: BLE001 - one bad plist skips one reading
            continue
        if watts is not None:
            out.append(watts)
    return out, tail


class PowerStream:
    def __init__(
        self,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._popen = popen
        self._clock = clock
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last: tuple[float, float] | None = None

    def _set(self, watts: float) -> None:
        with self._lock:
            self._last = (self._clock(), watts)

    def _read_loop(self, proc: subprocess.Popen) -> None:
        buf = b""
        try:
            while True:
                chunk = proc.stdout.read1(CHUNK_BYTES)
                if not chunk:
                    return
                readings, buf = parse_plists(buf + chunk)
                if readings:
                    self._set(readings[-1])
        except Exception:  # noqa: BLE001 - a dead stream just means no power
            return

    def start(self) -> bool:
        """True when powermetrics is running; False when it is not permitted or exits at once.

        Blocks for up to START_GRACE_S before the run starts, on purpose: that is how an
        immediate sudo refusal is noticed.
        """
        try:
            self._proc = self._popen(
                POWERMETRICS_ARGV,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False
        self._thread = threading.Thread(
            target=self._read_loop, args=(self._proc,), name="bakeoff-power", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + START_GRACE_S
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                return False
            time.sleep(0.05)
        return True

    def latest(self, max_age_s: float = 2.0) -> float | None:
        with self._lock:
            last = self._last
        if last is None or self._clock() - last[0] > max_age_s:
            return None
        return last[1]

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            proc.wait(STOP_TIMEOUT_S)  # reap, so no zombie is left
        except subprocess.TimeoutExpired:
            pass
        if proc.stdout is not None:
            proc.stdout.close()
        if self._thread is not None:
            self._thread.join(1.0)
