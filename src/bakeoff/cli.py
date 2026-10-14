"""Command line: run, env, verify, judge."""

import argparse
import asyncio
import getpass
import json
import os
import sys
import tomllib
from pathlib import Path

from bakeoff.bundle import verify_bundle
from bakeoff.client import EndpointClient
from bakeoff.config import ConfigError, RunConfig, load_config
from bakeoff.env import capture_env
from bakeoff.judge import JudgeError, Prices, run_judge
from bakeoff.runner import PASSES, PreflightError, execute_run, preflight
from bakeoff.suites import SuiteError

DEFAULT_DENYLIST = Path("~/.config/bakeoff/denylist")


def make_client(cfg: RunConfig) -> EndpointClient:
    key_env = cfg.endpoint.api_key_env
    api_key = os.environ.get(key_env) if key_env else None
    return EndpointClient(
        cfg.endpoint.base_url, cfg.endpoint.model, api_key=api_key, timeout_s=cfg.endpoint.timeout_s
    )


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(Path(args.config), args.overrides)
    except ConfigError as e:
        _err(str(e))
        return 1

    async def go() -> Path:
        async with make_client(cfg) as client:
            return await execute_run(
                cfg,
                client,
                Path(args.out),
                only=set(args.only) if args.only else None,
                extra_suites=args.suite,
            )

    try:
        path = asyncio.run(go())
    except (KeyboardInterrupt, asyncio.CancelledError):
        _err("interrupted; an incomplete bundle was written")
        return 130
    except (PreflightError, SuiteError, ConfigError) as e:
        _err(str(e))
        return 1
    except Exception as e:  # noqa: BLE001 - one line, no traceback unless debugging
        if os.environ.get("BAKEOFF_DEBUG") == "1":
            raise
        _err(f"unexpected {type(e).__name__}: {e}")
        return 1
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    print(path)
    if meta["status"] != "complete":
        _err(f"run incomplete: {meta['status_reason']}")
        return 2
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    cfg = info = served = None
    if args.config:
        try:
            cfg = load_config(Path(args.config))
        except ConfigError as e:
            _err(str(e))
            return 1

        async def probe():
            async with make_client(cfg) as client:
                try:
                    name = await preflight(client, cfg.endpoint.model)
                except PreflightError:
                    name = None
                return await client.runtime_info(cfg.model.runtime), name

        info, served = asyncio.run(probe())
    print(json.dumps(capture_env(cfg, info, served), indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    denylist = Path(args.denylist) if args.denylist else DEFAULT_DENYLIST.expanduser()
    findings = verify_bundle(Path(args.dir), getpass.getuser(), denylist)
    for f in findings:
        print(f"{f.file}:{f.line} {f.kind} {f.detail}")
    return 1 if findings else 0


def make_judge_client(base_url: str, model: str, api_key: str) -> EndpointClient:
    return EndpointClient(base_url, model, api_key=api_key)


def _judge_settings(args: argparse.Namespace) -> dict:
    table: dict = {}
    if args.judge_config:
        try:
            raw = tomllib.loads(Path(args.judge_config).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise JudgeError(f"cannot read judge config: {type(e).__name__}") from e
        table = raw.get("judge", {})
        if not isinstance(table, dict):
            raise JudgeError("[judge] must be a table")
    pick = lambda flag, key: flag if flag is not None else table.get(key)  # noqa: E731
    got = {
        "base_url": pick(args.judge_base_url, "base_url"),
        "model": pick(args.judge_model, "model"),
        "price_in": pick(args.price_in, "price_in"),
        "price_out": pick(args.price_out, "price_out"),
    }
    names = {
        "base_url": "judge base URL (--judge-base-url)",
        "model": "judge model (--judge-model)",
        "price_in": "input price (--price-in)",
        "price_out": "output price (--price-out)",
    }
    for k, v in got.items():
        if v is None or v == "":
            raise JudgeError(f"missing {names[k]}")
    for k in ("price_in", "price_out"):
        v = got[k]
        if isinstance(v, bool) or not isinstance(v, int | float) or not v >= 0 or v == float("inf"):
            raise JudgeError(f"{names[k]} must be a non-negative number")
    if not isinstance(got["base_url"], str) or not isinstance(got["model"], str):
        raise JudgeError("judge base URL and model must be strings")
    if not args.max_usd >= 0 or args.max_usd == float("inf"):
        raise JudgeError("--max-usd must be a non-negative number")
    return got


def _cmd_judge(args: argparse.Namespace) -> int:
    try:
        s = _judge_settings(args)
        key = os.environ.get("BAKEOFF_JUDGE_API_KEY")
        if not key:
            raise JudgeError("BAKEOFF_JUDGE_API_KEY is not set")
        prices = Prices(float(s["price_in"]), float(s["price_out"]))

        async def go() -> Path:
            async with make_judge_client(s["base_url"], s["model"], key) as client:
                return await run_judge(
                    Path(args.a),
                    Path(args.b),
                    client,
                    s["model"],
                    prices,
                    args.max_usd,
                    args.allow_partial,
                    Path(args.out),
                )

        path = asyncio.run(go())
    except (KeyboardInterrupt, asyncio.CancelledError):
        _err("interrupted; an incomplete judge bundle was written")
        return 130
    except (JudgeError, SuiteError) as e:
        _err(str(e))
        return 1
    except Exception as e:  # noqa: BLE001 - one line, no traceback unless debugging
        if os.environ.get("BAKEOFF_DEBUG") == "1":
            raise
        _err(f"unexpected {type(e).__name__}")
        return 1
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    print(path)
    if meta["status"] != "complete":
        _err(f"judge incomplete: {meta['status_reason']}")
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bakeoff")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the passes against one endpoint")
    r.add_argument("config")
    r.add_argument("--only", action="append", choices=PASSES)
    r.add_argument("--suite", action="append", default=[])
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--out", default="runs")
    r.set_defaults(fn=_cmd_run)
    e = sub.add_parser("env", help="print the environment capture")
    e.add_argument("--config")
    e.set_defaults(fn=_cmd_env)
    v = sub.add_parser("verify", help="check hashes and scan for scrub violations")
    v.add_argument("dir")
    v.add_argument("--denylist")
    v.set_defaults(fn=_cmd_verify)
    j = sub.add_parser("judge", help="blind pairwise judge over two bundles")
    j.add_argument("a")
    j.add_argument("b")
    j.add_argument("--judge-model")
    j.add_argument("--max-usd", type=float, required=True)
    j.add_argument("--judge-base-url")
    j.add_argument("--price-in", type=float)
    j.add_argument("--price-out", type=float)
    j.add_argument("--judge-config")
    j.add_argument("--allow-partial", action="store_true")
    j.add_argument("--out", default="runs")
    j.set_defaults(fn=_cmd_judge)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
