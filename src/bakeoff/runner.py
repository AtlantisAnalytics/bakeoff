"""The runner: preflight, the three passes, sampling, redaction and the hashed bundle."""

import asyncio
import dataclasses
import ipaddress
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from bakeoff.bundle import run_id, write_bundle
from bakeoff.client import EndpointClient, normalize_base_url
from bakeoff.config import RunConfig, public_config
from bakeoff.env import Runner, capture_env, default_run, detect_platform, field
from bakeoff.errors import AbortRun
from bakeoff.fit import run_fit
from bakeoff.hashing import canonical_json
from bakeoff.powermetrics import PowerStream
from bakeoff.quality import QualityResult, run_quality
from bakeoff.sampler import Sampler
from bakeoff.speed import run_speed
from bakeoff.suites import Suite, check_lock, resolve_suites, suites_root

PASSES = ("quality", "speed", "fit")
REDACTED = "<endpoint>"
MAX_LISTED_MODELS = 10


class PreflightError(Exception):
    pass


async def preflight(client: EndpointClient, model: str) -> str:
    """Return the served name for `model`; Ollama's implicit ':latest' tag is accepted."""
    try:
        served = await client.list_models()
    except Exception as e:  # noqa: BLE001 - any failure to list models is a preflight failure
        raise PreflightError(f"cannot list models at the endpoint: {type(e).__name__}: {e}") from e
    for name in (model, f"{model}:latest", model.removesuffix(":latest")):
        if name in served:
            return name
    shown = ", ".join(served[:MAX_LISTED_MODELS]) or "(none)"
    more = f" (+{len(served) - MAX_LISTED_MODELS} more)" if len(served) > MAX_LISTED_MODELS else ""
    raise PreflightError(
        f"model {model!r} is not served by the endpoint. Served: {shown}{more}. "
        "Set endpoint.model to the served-model-name."
    )


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _endpoint_terms(base_url: str) -> tuple[list[str], list[str], list[str]]:
    """(always, anchored, bare).

    `always` strings are redacted wherever they occur; `anchored` host:port forms need a clean
    left edge (so "ollama:8000" survives host "llama"); `bare` hosts need both edges.
    """
    api_base, root = normalize_base_url(base_url)
    always = [base_url, base_url.strip(), base_url.strip().rstrip("/"), api_base, root]
    anchored: list[str] = []
    bare: list[str] = []
    try:
        parts = urlsplit(base_url if "//" in base_url else "//" + base_url)
        host, port, netloc = parts.hostname or "", parts.port, parts.netloc
    except ValueError:
        host, port, netloc = "", None, ""
    if "@" in netloc:
        always.append(netloc)  # userinfo
    if host and port is not None:
        anchored.append(f"{host}:{port}")
        anchored.append(f"[{host}]:{port}")
    if host and (_is_ip(host) or "." in host):
        bare.append(host)
        if ":" in host:
            bare.append(f"[{host}]")
    key = lambda t: -len(t)  # noqa: E731
    return (
        sorted({t for t in always if t}, key=key),
        sorted({t for t in anchored if t}, key=key),
        sorted({t for t in bare if t}, key=key),
    )


_PLACEHOLDER_KEYS = frozenset({"ollama", "empty", "none", "sk-no-key-required"})


def make_redactor(base_url: str, secrets: tuple[str, ...] = ()) -> Callable[[object], object]:
    always, anchored, bare = _endpoint_terms(base_url)
    keep = (s for s in secrets if len(s) >= 8 and s.lower() not in _PLACEHOLDER_KEYS)
    always = sorted({*always, *keep}, key=lambda t: -len(t))
    alts = [re.escape(t) for t in always]
    alts += [rf"(?<![\w.-]){re.escape(t)}" for t in anchored]
    alts += [rf"(?<![\w.-]){re.escape(t)}(?![\w.-])" for t in bare]
    pattern = re.compile("|".join(alts), re.I) if alts else None

    def redact(obj: object) -> object:
        if isinstance(obj, str):
            return pattern.sub(REDACTED, obj) if pattern else obj
        if isinstance(obj, dict):
            return {redact(k): redact(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [redact(v) for v in obj]
        return obj

    return redact


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _warn_lock(found: list[Suite]) -> None:
    root = suites_root().resolve()
    ours = [s for s in found if s.path.resolve().is_relative_to(root)]
    if not ours:
        return
    try:
        lock = json.loads((root / "SUITES.lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"warning: cannot read SUITES.lock.json: {e}", file=sys.stderr)
        return
    for problem in check_lock(ours, lock):
        print(f"warning: {problem}", file=sys.stderr)


def _config_for_bundle(cfg: RunConfig) -> dict:
    """public_config, with local suite paths replaced: paths never go in a bundle."""
    d = public_config(cfg)
    q = d["quality"]
    q["private_suites"] = ["<private>"] * len(q["private_suites"])
    q["suites"] = ["<path>" if Path(s).is_absolute() else s for s in q["suites"]]
    return d


def _load_suites(cfg: RunConfig, extra_suites: list[str]) -> tuple[list[Suite], set[str]]:
    private_keys = {s.key for s in resolve_suites(list(cfg.quality.private_suites))}
    suites = resolve_suites([*cfg.quality.suites, *cfg.quality.private_suites, *extra_suites])
    private_keys |= {s.key for s in suites if s.split == "private"}
    return suites, private_keys


_PATHLIKE = re.compile(r"/Users/|/home/|[A-Za-z]:\\|^/", re.I)


def public_name(value: str) -> str:
    """A path-like identifier is cut to its basename; anything else is returned unchanged."""
    if not _PATHLIKE.search(value):
        return value
    return re.split(r"[\\/]+", value.rstrip("/\\"))[-1] or "<path>"


def _sanitize_cfg(cfg: RunConfig) -> RunConfig:
    """Replace path-like model identifiers with basenames (one stderr warning per field)."""
    changes: dict[str, dict[str, str]] = {"endpoint": {}, "model": {}}
    for section, key in (("endpoint", "model"), ("model", "weights_file"), ("model", "label")):
        value = getattr(getattr(cfg, section), key)
        clean = public_name(value)
        if clean != value:
            changes[section][key] = clean
            print(
                f"warning: {section}.{key} looks like a path; the bundle keeps only {clean!r}",
                file=sys.stderr,
            )
    return dataclasses.replace(
        cfg,
        endpoint=dataclasses.replace(cfg.endpoint, **changes["endpoint"]),
        model=dataclasses.replace(cfg.model, **changes["model"]),
    )


REPORT_TIMEOUT_S = 10.0


async def _runtime_reported(client: EndpointClient, runtime: str) -> dict:
    # No live KV gauge: read after the run, the cache is idle and it always says 0.0. Fit's KV
    # capacity and per-step fractions (fit.kv) are the KV facts.
    size_vram = None
    try:
        info = await asyncio.wait_for(client.runtime_info(runtime), REPORT_TIMEOUT_S)
        size_vram = info.get("size_vram")
    except Exception:  # noqa: BLE001 - memory facts are best effort
        pass
    return {"size_vram": field(size_vram, "reported")}


async def _fit_probe(client: EndpointClient, runtime: str, first: dict) -> tuple[dict, str | None]:
    """Probe again right before the fit pass: Ollama only reports context for a loaded model."""
    try:
        again = await asyncio.wait_for(client.runtime_info(runtime), REPORT_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - fall back to the first probe
        again = {}
    if again.get("context_length"):
        return again, "pre_fit"
    if first.get("context_length"):
        return first, "initial"
    return first, None


async def execute_run(
    cfg: RunConfig,
    client: EndpointClient,
    out_root: Path,
    only: set[str] | None = None,
    extra_suites: list[str] = (),  # type: ignore[assignment]
    run: Runner = default_run,
    platform: str = sys.platform,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sample_interval_s: float = 0.5,
) -> Path:
    wanted = set(PASSES) if only is None else set(only)
    served = await preflight(client, cfg.endpoint.model)
    runtime_name = cfg.model.runtime
    info = await client.runtime_info(runtime_name)
    key_env = cfg.endpoint.api_key_env
    secrets = (os.environ.get(key_env, ""),) if key_env else ()
    cfg = _sanitize_cfg(cfg)  # after preflight, which needs the real served name
    env = capture_env(cfg, info, public_name(served), run=run, platform=platform)
    suites: list[Suite] = []
    private_keys: set[str] = set()
    if "quality" in wanted:
        suites, private_keys = _load_suites(cfg, list(extra_suites))
        _warn_lock(suites)

    started = now()
    started_at = _iso(started)
    pub = _config_for_bundle(cfg)
    salt = canonical_json(public_config(cfg)) + started_at.encode() + os.urandom(16)
    rid = run_id(
        started, cfg.model.label, salt, "-".join(sorted(wanted)) if wanted != set(PASSES) else ""
    )

    sampler_platform = detect_platform(run, platform)
    power = PowerStream() if sampler_platform == "darwin" else None
    sampler = Sampler(sampler_platform, run, interval_s=sample_interval_s, power=power)
    quality = QualityResult()
    speed_out: tuple[list[dict], dict] | None = None
    fit_out: tuple[list[dict], dict] | None = None
    status, reason = "complete", None
    interrupted: BaseException | None = None
    try:
        if power is not None:
            power.start()  # False (not permitted) just leaves power unrecorded
        sampler.start()
        if "quality" in wanted:
            sampler.mark("quality")
            try:
                quality = await run_quality(client, suites)
            except (AbortRun, asyncio.CancelledError, KeyboardInterrupt) as exc:
                quality = getattr(exc, "partial", quality)
                raise
        if "speed" in wanted:
            try:
                speed_out = await run_speed(client, cfg.speed, runtime_name, sampler)
            except (AbortRun, asyncio.CancelledError, KeyboardInterrupt) as exc:
                speed_out = getattr(exc, "partial", speed_out)
                raise
        if "fit" in wanted:
            try:
                fit_info, probe = await _fit_probe(client, runtime_name, info)
                fit_out = await run_fit(
                    client, cfg.fit, runtime_name, fit_info.get("context_length"), sampler, probe
                )
            except (AbortRun, asyncio.CancelledError, KeyboardInterrupt) as exc:
                fit_out = getattr(exc, "partial", fit_out)
                raise
    except AbortRun as exc:
        status, reason = "incomplete", exc.reason
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        status, reason = "incomplete", "interrupted"
        interrupted = exc
    finally:
        try:
            samples = sampler.stop()
        finally:  # powermetrics runs as root: it must never outlive the run
            if power is not None:
                power.stop()

    reported = (
        {"size_vram": field(None, "reported")}
        if interrupted
        else await _runtime_reported(client, runtime_name)
    )
    peaks = sampler.peaks()
    files: dict[str, list[dict]] = {}
    for key, recs in quality.records.items():
        suite = next(s for s in suites if s.key == key)
        folder = "private" if key in private_keys else "quality"
        files[f"{folder}/{suite.track}.{suite.name}.jsonl"] = recs
    if speed_out is not None:
        files["speed.jsonl"] = speed_out[0]
    if fit_out is not None:
        files["fit.jsonl"] = fit_out[0]
    files["samples.jsonl"] = samples

    meta = {
        "bundle_format": "m2",
        "run_id": rid,
        "started_at": started_at,
        "ended_at": _iso(now()),
        "status": status,
        "status_reason": reason,
        "config": pub,
        "env": env,
        "suites": [
            {"key": s.key, "version": s.version, "split": s.split, "hash": s.hash} for s in suites
        ],
        "quality": {"suites": quality.suites, "tracks": quality.tracks},
        "speed": speed_out[1] if speed_out else {},
        "fit": fit_out[1] if fit_out else {},
        "memory": {
            "platform": field(sampler.platform, "measured"),
            "baseline": field(peaks.get("baseline"), "measured"),
            "peaks": field(peaks or None, "measured"),
            "runtime_reported": reported,
        },
    }
    redact = make_redactor(cfg.endpoint.base_url, secrets)
    meta = redact(meta)  # type: ignore[assignment]
    files = {rel: redact(recs) for rel, recs in files.items()}  # type: ignore[misc]
    run_dir = Path(out_root) / rid
    write_bundle(run_dir, meta, files)
    if interrupted is not None:
        interrupted.bundle_written = True  # type: ignore[attr-defined]
        raise interrupted
    return run_dir
