"""Command line: run, env, verify."""

import argparse
import asyncio
import getpass
import json
import os
import sys
from pathlib import Path

from bakeoff.bundle import verify_bundle
from bakeoff.client import EndpointClient
from bakeoff.config import ConfigError, RunConfig, load_config
from bakeoff.env import capture_env
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
