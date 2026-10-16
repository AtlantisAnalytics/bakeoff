"""Background memory and power sampler.

Peaks are published per pass, so the arithmetic is exact: totals are the max of per-sample sums,
and power is aggregated only over samples where every GPU reported a number. The sampler never
raises into the run: a failed or malformed reading just skips that tick.
"""

import csv
import math
import threading
import time

from bakeoff.env import Runner, nvidia_argv, parse_vm_stat

SAMPLER_NVIDIA_ARGV = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,power.draw",
    "--format=csv,noheader,nounits",
]
DARWIN_ARGV = ["vm_stat"]
JOIN_TIMEOUT_S = 5.0


def parse_sampler_query(text: str) -> list[dict] | None:
    """Parse the 3-column sampler query. Any malformed row spoils the whole sample (None)."""
    gpus = []
    for rec in csv.reader(text.splitlines(), skipinitialspace=True):
        if not rec:
            continue
        if len(rec) != 3:
            return None
        idx, used, draw = (c.strip() for c in rec)
        try:
            index, mem = int(idx), int(used)
        except ValueError:
            return None
        gpus.append({"index": index, "mem_used_mib": mem, "power_w": _power(draw)})
    return gpus or None


def _power(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:  # "[N/A]", "[Not Supported]", junk
        return None
    return value if math.isfinite(value) else None


class Sampler:
    def __init__(self, platform: str, run: Runner, interval_s: float = 0.5):
        self.platform = platform
        self._run = run
        self._interval = interval_s
        self._samples: list[dict] = []
        self._lock = threading.Lock()
        self._pass = "baseline"
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def captured(self) -> bool:
        with self._lock:
            return bool(self._samples)

    def mark(self, pass_name: str) -> None:
        with self._lock:
            self._pass = pass_name

    def _read(self) -> dict | None:
        """One reading without t/pass, or None."""
        if self.platform == "nvidia":
            gpus = parse_sampler_query(self._run(nvidia_argv(SAMPLER_NVIDIA_ARGV)) or "")
            return None if gpus is None else {"gpus": gpus}
        if self.platform == "darwin":
            used = parse_vm_stat(self._run(DARWIN_ARGV) or "")
            return None if used is None else {"mem_used_bytes": used}
        return None

    def sample_once(self) -> dict | None:
        try:
            with self._lock:  # label and time belong to the moment the read starts
                label, t = self._pass, time.monotonic() - self._t0
            reading = self._read()
            if reading is None:
                return None
            with self._lock:
                if self._pass != label:  # the pass changed mid-read: the label would be wrong
                    return None
                rec = {"t": t, "pass": label, **reading}
                self._samples.append(rec)
            return rec
        except Exception:  # noqa: BLE001 - the sampler must never kill a run
            return None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample_once()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._t0 = time.monotonic()
        self.sample_once()
        self._thread = threading.Thread(target=self._loop, name="bakeoff-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(JOIN_TIMEOUT_S)
        with self._lock:
            return list(self._samples)

    def peaks(self) -> dict[str, dict]:
        with self._lock:
            samples = list(self._samples)
        by_pass: dict[str, list[dict]] = {}
        for s in samples:
            by_pass.setdefault(s["pass"], []).append(s)
        return {name: self._peak(rows) for name, rows in by_pass.items()}

    def _peak(self, rows: list[dict]) -> dict:
        if self.platform == "darwin":
            return {"mem_used_bytes": max(r["mem_used_bytes"] for r in rows)}
        per_gpu: dict[int, int] = {}
        for r in rows:
            for g in r["gpus"]:
                per_gpu[g["index"]] = max(
                    per_gpu.get(g["index"], g["mem_used_mib"]), g["mem_used_mib"]
                )
        sums = [
            sum(g["power_w"] for g in r["gpus"])
            for r in rows
            if all(g["power_w"] is not None for g in r["gpus"])
        ]
        return {
            "mem_used_mib_total": max(sum(g["mem_used_mib"] for g in r["gpus"]) for r in rows),
            "mem_used_mib_per_gpu": [per_gpu[i] for i in sorted(per_gpu)],
            "power_w_mean": sum(sums) / len(sums) if sums else None,
            "power_w_max": max(sums) if sums else None,
        }
