"""Read-only view over one run bundle, plus the eligibility check."""

import json
from dataclasses import dataclass, field
from pathlib import Path


class RunViewError(ValueError):
    pass


@dataclass(frozen=True)
class RunView:
    run_id: str
    bundle_sha256: str
    path: Path
    bundle_format: str
    status: str
    envelope: str
    model: dict
    tracks: dict[str, float | None]
    suites: dict[str, dict]
    items: dict[str, dict[str, bool]]
    short_decode: float | None
    max_context: int | None
    speed: dict
    fit: dict
    size_vram: int | None
    harness: dict
    meta: dict = field(repr=False)


def _obj(parent: object, key: str) -> dict:
    v = parent.get(key) if isinstance(parent, dict) else None
    return v if isinstance(v, dict) else {}


def _value(parent: object, key: str):
    v = _obj(parent, key)
    return v.get("value")


def _read_items(path: Path, key: str) -> dict[str, bool]:
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise RunViewError(f"cannot read item records for {key}: {e}") from e
    out: dict[str, bool] = {}
    for no, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except (ValueError, RecursionError) as e:
            raise RunViewError(f"{path.name} line {no}: not valid JSON") from e
        if not isinstance(rec, dict) or not isinstance(rec.get("id"), str):
            raise RunViewError(f"{path.name} line {no}: record has no id")
        out[rec["id"]] = rec.get("pass") is True
    return out


def load_run(path: Path) -> RunView:
    path = Path(path)
    meta_path = path / "bundle.json"
    try:
        meta = json.loads(meta_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError, RecursionError) as e:
        raise RunViewError(f"cannot read {meta_path.name} in {path.name}: {e}") from e
    if not isinstance(meta, dict):
        raise RunViewError("bundle.json is not an object")
    for req in ("run_id", "bundle_sha256", "status"):
        if not isinstance(meta.get(req), str):
            raise RunViewError(f"bundle.json lacks {req}")
    model = _obj(_obj(meta, "config"), "model")
    quality = _obj(meta, "quality")
    summaries = quality.get("suites")
    if not isinstance(summaries, list):
        raise RunViewError("bundle.json lacks quality.suites")
    suites: dict[str, dict] = {}
    items: dict[str, dict[str, bool]] = {}
    for s in summaries:
        if not isinstance(s, dict) or not all(
            isinstance(s.get(k), str) for k in ("key", "track", "name")
        ):
            raise RunViewError("malformed suite summary")
        suites[s["key"]] = s
        if s.get("accuracy") is None:  # judge-scored: no per-item verdicts
            continue
        folder = "private" if s.get("split") == "private" else "quality"
        items[s["key"]] = _read_items(path / folder / f"{s['track']}.{s['name']}.jsonl", s["key"])
    speed = _obj(meta, "speed")
    fit = _obj(meta, "fit")
    decode = _obj(speed.get("short@1"), "decode_tps").get("median")
    ctx = fit.get("max_context_passed")
    harness = _obj(_obj(meta, "env"), "harness")
    tracks = quality.get("tracks")
    return RunView(
        run_id=meta["run_id"],
        bundle_sha256=meta["bundle_sha256"],
        path=path,
        bundle_format=str(meta.get("bundle_format", "")),
        status=meta["status"],
        envelope=str(model.get("envelope", "")),
        model={k: model.get(k) for k in ("label", "weights", "quant", "runtime")},
        tracks=dict(tracks) if isinstance(tracks, dict) else {},
        suites=suites,
        items=items,
        short_decode=decode if isinstance(decode, (int, float)) else None,
        max_context=ctx if isinstance(ctx, int) and not isinstance(ctx, bool) else None,
        speed=speed,
        fit=fit,
        size_vram=_value(_obj(_obj(meta, "memory"), "runtime_reported"), "size_vram"),
        harness={
            "version": _value(harness, "version"),
            "git_commit": _value(harness, "git_commit"),
        },
        meta=meta,
    )


def model_key(run: RunView) -> tuple[str, str, str]:
    m = run.model
    return (m["weights"], m["quant"], m["runtime"])


def ineligible_reasons(run: RunView, rules: dict) -> list[str]:
    reasons: list[str] = []
    if run.status != "complete":
        reasons.append("status_incomplete")
    if run.max_context is None:
        reasons.append("no_fit")
    elif run.max_context < rules["min_context"]:
        reasons.append("context_below_min")
    if run.short_decode is None:
        reasons.append("no_speed")
    elif run.short_decode < rules["min_decode_tps"]:
        reasons.append("decode_below_min")
    return reasons
