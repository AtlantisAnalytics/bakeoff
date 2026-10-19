"""Fixture builder: writes a real, verifiable run bundle with fictional values."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bakeoff.bundle import run_id as make_run_id
from bakeoff.bundle import write_bundle
from bakeoff.hashing import sha256_hex

_STARTED = datetime(2026, 10, 12, 9, 0, 0, tzinfo=UTC)


def _ci(acc: float) -> list[float]:
    return [max(0.0, acc - 0.1), min(1.0, acc + 0.1)]


def make_bundle(
    dir: Path,
    *,
    label: str,
    weights: str,
    quant: str = "Q4_K_M",
    runtime: str = "ollama",
    envelope: str,
    grades: dict[str, list[bool]],
    suite_hashes: dict[str, str] | None = None,
    decode: float = 40.0,
    max_context: int = 32768,
    status: str = "complete",
    fmt: str = "m2",
    speed: bool = True,
    fit: bool = True,
    run_id: str | None = None,
    private: tuple[str, ...] = (),
    judge: tuple[str, ...] = (),
    harness_version: str = "0.1.0",
    git_commit: str = "0123456789abcdef0123456789abcdef01234567",
    size_vram: int | None = 8_000_000_000,
    fit_extra: dict | None = None,
) -> Path:
    """Write a bundle under `dir` (created) and return its directory.

    `grades` maps a suite key ("general/gsm8k") to per-item results, ids "<name>-<i>".
    `private` lists suite keys stored under private/; `judge` lists suite keys that are
    judge-scored (no pass/accuracy, so they carry no item verdicts).
    """
    dir = Path(dir)
    if run_id is None:
        salt = json.dumps([label, weights, quant, runtime, envelope, status, fmt], sort_keys=True)
        run_id = make_run_id(_STARTED, label, salt.encode())
    hashes = suite_hashes or {}
    files: dict[str, list[dict]] = {}
    suites: list[dict] = []
    by_track: dict[str, list[float]] = {}
    all_keys = [*grades, *[k for k in judge if k not in grades]]
    for key in all_keys:
        track, name = key.split("/", 1)
        is_judge = key in judge
        results = [] if is_judge else grades[key]
        split = "private" if key in private else "public"
        recs = []
        for i, ok in enumerate(results):
            rec = {"id": f"{name}-{i}", "pass": ok, "error": None}
            if fmt == "m2" and track == "structured":
                rec["extractable"] = ok
            recs.append(rec)
        if is_judge:
            recs = [{"id": f"{name}-{i}", "pass": None, "error": None} for i in range(3)]
        folder = "private" if split == "private" else "quality"
        files[f"{folder}/{track}.{name}.jsonl"] = recs
        n = len(recs)
        passed = None if is_judge else sum(1 for r in recs if r["pass"] is True)
        acc = None if is_judge or n == 0 else passed / n
        h = hashes.get(key) or sha256_hex(key.encode())
        summary = {
            "key": key,
            "track": track,
            "name": name,
            "version": 1,
            "split": split,
            "hash": h,
            "n": n,
            "passed": passed,
            "errors": 0,
            "truncated": 0,
            "accuracy": acc,
            "ci95": None if acc is None else _ci(acc),
        }
        if fmt == "m2" and track == "structured" and acc is not None:
            summary["extractable_accuracy"] = acc
            summary["extractable_ci95"] = _ci(acc)
        suites.append(summary)
        if acc is not None:
            by_track.setdefault(track, []).append(acc)
    tracks = {t: sum(v) / len(v) for t, v in sorted(by_track.items())}
    for s in suites:
        tracks.setdefault(s["track"], None)

    def field(v, src="measured"):
        return {"value": v, "source": src}

    meta: dict = {
        "bundle_format": fmt,
        "run_id": run_id,
        "started_at": "2026-10-12T09:00:00Z",
        "ended_at": "2026-10-12T10:00:00Z",
        "status": status,
        "status_reason": None if status == "complete" else "interrupted",
        "config": {
            "model": {
                "label": label,
                "weights": weights,
                "quant": quant,
                "runtime": runtime,
                "envelope": envelope,
            },
        },
        "env": {
            "harness": {
                "version": field(harness_version),
                "git_commit": field(git_commit),
            },
        },
        "suites": [{k: s[k] for k in ("key", "version", "split", "hash")} for s in suites],
        "quality": {"suites": suites, "tracks": tracks},
        "memory": {"runtime_reported": {"size_vram": field(size_vram, "reported")}},
    }
    if speed:
        stat = {"median": decode, "min": decode, "max": decode, "n": 5}
        meta["speed"] = {"short@1": {"decode_tps": stat, "usage_source": "reported"}}
        files["speed.jsonl"] = [{"cell": "short@1", "decode_tps": decode}]
    if fit:
        meta["fit"] = {"max_context_passed": max_context, "server_context": 65536}
        meta["fit"].update(fit_extra or {})
        files["fit.jsonl"] = [{"step": max_context, "status": "pass"}]
    out = dir / run_id
    write_bundle(out, meta, files)
    return out
