"""Command line: run, env, verify, judge, publish, site, rederive."""

import argparse
import asyncio
import getpass
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

from bakeoff.bundle import GENERIC_USERNAMES, verify_bundle
from bakeoff.client import EndpointClient
from bakeoff.config import ConfigError, RunConfig, load_config
from bakeoff.env import capture_env
from bakeoff.hashing import bad_relpath
from bakeoff.judge import JudgeError, Prices, run_judge
from bakeoff.ledger import LedgerError, bundle_kind, load_layout, week_of
from bakeoff.publish import PublishError, publish, week_docs, write_picks
from bakeoff.rederive import RederiveError, rederive
from bakeoff.runner import PASSES, PreflightError, execute_run, preflight
from bakeoff.s3 import S3Client
from bakeoff.store import HashMismatch, Store, StoreError, materialize, object_key
from bakeoff.suites import SuiteError, resolve_suites, suites_root
from bakeoff.web.build import check_site, render_site, write_preview, write_site
from bakeoff.weekcfg import WeekInputError

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

    written = {"bundle": False}

    async def go() -> Path:
        async with make_client(cfg) as client:
            try:
                return await execute_run(
                    cfg,
                    client,
                    Path(args.out),
                    only=set(args.only) if args.only else None,
                    extra_suites=args.suite,
                )
            except (KeyboardInterrupt, asyncio.CancelledError) as e:
                # asyncio.run may re-raise a fresh KeyboardInterrupt, so note it here
                written["bundle"] = bool(getattr(e, "bundle_written", False))
                raise

    try:
        path = asyncio.run(go())
    except (KeyboardInterrupt, asyncio.CancelledError):
        _err(
            "interrupted; an incomplete bundle was written"
            if written["bundle"]
            else "interrupted; no bundle was written"
        )
        return 130
    except (PreflightError, SuiteError, ConfigError, FileExistsError) as e:
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


def _public_suite_text() -> str:
    """Prompt and expected text of the shipped suites (username false-positive check)."""
    try:
        suites = resolve_suites([str(suites_root())])
    except Exception:
        return ""
    parts: list[str] = []
    for s in suites:
        for item in s.items:
            parts.extend(str(m.get("content", "")) for m in item.messages)
            parts.append(json.dumps(item.expected, ensure_ascii=False))
    return "\n".join(parts)


def pick_username(
    no_username: bool, public_text=_public_suite_text
) -> tuple[str | None, str | None, bool]:
    """(username to scan for, one-line notice, skip_output).

    A None name means the username check is skipped. skip_output means the name is an ordinary
    word in the public suites, so model `output` fields are exempt from the scan.
    """
    if no_username:
        return None, None, False
    try:
        name = getpass.getuser()
    except (OSError, KeyError, ImportError):  # e.g. a container with an arbitrary UID
        return None, None, False
    if not name or name.lower() in GENERIC_USERNAMES or len(name) <= 3:
        return None, None, False
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", public_text(), re.I):
        return (
            name,
            (
                f"notice: {name!r} is an ordinary word in the public suites, so the username "
                "check skips model output fields; use --denylist for stronger protection"
            ),
            True,
        )
    return name, None, False


def _cmd_verify(args: argparse.Namespace) -> int:
    denylist = Path(args.denylist) if args.denylist else DEFAULT_DENYLIST.expanduser()
    username, notice, skip_output = pick_username(args.no_username)
    if notice:
        print(notice, file=sys.stderr)
    run_dir = Path(args.dir)
    if (run_dir / "private").exists():
        print("notice: contains private-suite outputs; do not share", file=sys.stderr)
    extra: dict = {}
    try:
        extra = _split_context(run_dir)
    except (LedgerError, StoreError, OSError) as e:
        _err(str(e))
        return 1
    findings = verify_bundle(run_dir, username, denylist, username_skip_output=skip_output, **extra)
    for f in findings:
        print(f"{f.file}:{f.line} {f.kind} {f.detail}")
    return 1 if findings else 0


def _split_context(run_dir: Path) -> dict:
    """kept/store arguments for a bundle under results/<week>/(runs|judge)/<id> in a split range."""
    resolved = run_dir.resolve()
    if len(resolved.parts) < 5:
        return {}
    root = resolved.parents[3]
    week = week_of(root, resolved)
    if week is None:
        return {}
    try:
        meta = json.loads((resolved / "bundle.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(meta, dict):
        return {}
    kept = load_layout(root).kept(week, bundle_kind(meta))
    if kept is None:
        return {}
    return {"kept": kept, "store": Store.from_env(root)}


def make_judge_client(base_url: str, model: str, api_key: str, transport=None) -> EndpointClient:
    # no retries: each judge call is exactly one billed attempt, so the budget math holds
    return EndpointClient(base_url, model, api_key=api_key, backoff_s=(), transport=transport)


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


def _cmd_publish(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if args.denylist:
        denylist = Path(args.denylist)
    else:
        default = DEFAULT_DENYLIST.expanduser()
        denylist = default if default.is_file() else None
    username, notice, skip_output = pick_username(args.no_username)
    if notice:
        print(notice, file=sys.stderr)
    try:
        out = publish(
            Path(args.week_toml),
            root,
            username=username,
            denylist=denylist,
            username_skip_output=skip_output,
        )
        print(f"published week {out.name}")
        write_site(render_site(root), root / "site")
    except (PublishError, WeekInputError, OSError, ValueError) as e:
        _err(str(e))
        return 1
    return 0


def _cmd_site(args: argparse.Namespace) -> int:
    root = Path(args.root)
    try:
        if args.check:
            problems = check_site(root)
            for line in problems:
                print(line)
            if problems:
                _err("site/ is out of date; run `bakeoff site`")
            return 1 if problems else 0
        if args.preview:
            out = Path(args.preview)
            if out.resolve().is_relative_to((root / "site").resolve()):
                _err("--preview must not be site/ or a path inside it")
                return 2
            write_preview(root, out)
            return 0
        if week_docs(root) or (root / "results" / "picks.json").exists():
            write_picks(root)
        write_site(render_site(root), root / "site")
    except (PublishError, WeekInputError, OSError, ValueError) as e:
        _err(str(e))
        return 1
    return 0


def _cmd_rederive(args: argparse.Namespace) -> int:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.week):
        _err(f"week must be YYYY-MM-DD, got {args.week!r}")
        return 2
    try:
        problems = rederive(Path(args.root), args.week)
    except (RederiveError, OSError) as e:
        _err(str(e))
        return 1
    if not problems:
        print("identical")
        return 0
    print(f"differs: results/{args.week}/week.json does not match its re-derivation")
    for line in problems:
        print(f"  {line}")
    return 1


def _listed(bundle: Path) -> tuple[bytes, list[tuple[str, str]]]:
    try:
        raw = (bundle / "bundle.json").read_bytes()
        meta = json.loads(raw)
    except (OSError, ValueError) as e:
        raise StoreError(f"{bundle.name}: cannot read bundle.json: {type(e).__name__}") from None
    files = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files, dict):
        raise StoreError(f"{bundle.name}: bundle.json has no files object")
    for rel, sha in files.items():
        if bad_relpath(rel) or not isinstance(sha, str):
            raise StoreError(f"{bundle.name}: bad entry in files: {rel!r}")
    return raw, sorted(files.items())


def _store_each(args: argparse.Namespace, one) -> int:
    code = 0
    try:
        store = Store.from_env(Path(args.root))
        for b in args.bundles:
            try:
                if one(Path(b), store):
                    code = 1
            except StoreError as e:
                print(f"{Path(b).name}: {e}")
                code = 1
    except (StoreError, LedgerError, OSError) as e:
        _err(str(e))
        return 1
    return code


def _cmd_store_fetch(args: argparse.Namespace) -> int:
    def one(bundle: Path, store: Store) -> bool:
        _, listed = _listed(bundle)
        fetched = cached = 0
        for rel, sha in listed:
            if (bundle / rel).is_file():
                cached += 1
            elif store.cached(sha):
                store.resolve(sha)
                cached += 1
            else:
                store.resolve(sha)
                fetched += 1
        if args.into:
            materialize(bundle, Path(args.into) / bundle.name, store)
        print(f"{bundle.name}: fetched {fetched}, cached {cached}")
        return False

    return _store_each(args, one)


def _cmd_store_push(args: argparse.Namespace) -> int:
    if args.denylist:
        denylist = Path(args.denylist)
    else:
        default = DEFAULT_DENYLIST.expanduser()
        denylist = default if default.is_file() else None
    username, notice, skip_output = pick_username(args.no_username)
    if notice:
        print(notice, file=sys.stderr)
    try:
        client = S3Client.from_env()
    except StoreError as e:
        _err(str(e))
        return 1
    code = 0
    for b in args.bundles:
        bundle = Path(b)
        try:
            _, listed = _listed(bundle)
            findings = verify_bundle(
                bundle, username, denylist, username_skip_output=skip_output, kept=None
            )
            if findings:
                for f in findings:
                    print(f"{f.file}:{f.line} {f.kind} {f.detail}")
                print(f"{bundle.name}: refusing to push a bundle that does not verify")
                code = 1
                continue
            uploaded = skipped = 0
            for rel, sha in listed:
                key = object_key(sha)
                if client.head(key):
                    skipped += 1
                else:
                    client.put(key, bundle / rel, sha)
                    uploaded += 1
            print(f"{bundle.name}: uploaded {uploaded}, skipped {skipped}")
        except (StoreError, OSError) as e:
            print(f"{bundle.name}: {e}")
            code = 1
    return code


def _cmd_store_check(args: argparse.Namespace) -> int:
    def one(bundle: Path, store: Store) -> bool:
        raw, listed = _listed(bundle)
        bad = False
        ok = 0
        present: set[str] = set()
        for rel, sha in listed:
            if sha in store.withdrawn:
                print(f"withdrawn {sha} {store.withdrawn[sha].date}")
            elif store.exists(sha):
                ok += 1
                present.add(sha)
            else:
                print(f"missing {sha} {rel}")
                bad = True
        if args.rehash and listed:
            rng = random.Random(hashlib.sha256(raw).hexdigest())
            chosen = rng.sample(listed, min(args.rehash, len(listed)))
            tmp = Path(tempfile.mkdtemp(prefix="bakeoff-rehash-"))
            try:
                fresh = Store(store.base_url, tmp, {})
                for rel, sha in chosen:
                    if sha not in present:
                        continue
                    try:
                        fresh.resolve(sha)
                    except HashMismatch:
                        print(f"corrupt {sha} {rel}")
                        bad = True
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if not bad:
            print(f"{bundle.name}: ok {ok}")
        return bad

    return _store_each(args, one)


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
    v.add_argument(
        "--no-username", action="store_true", help="do not scan for the current username"
    )
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
    b = sub.add_parser("publish", help="publish one week and regenerate site/")
    b.add_argument("week_toml")
    b.add_argument("--root", default=".")
    b.add_argument("--denylist")
    b.add_argument(
        "--no-username", action="store_true", help="do not scan for the current username"
    )
    b.set_defaults(fn=_cmd_publish)
    st = sub.add_parser("site", help="regenerate site/ from results/ and site.toml")
    st.add_argument("--root", default=".")
    mode = st.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 if site/ is out of date")
    mode.add_argument("--preview", metavar="DIR", help="build a launched site into DIR")
    st.set_defaults(fn=_cmd_site)
    rd = sub.add_parser("rederive", help="re-publish a published week and compare its week.json")
    rd.add_argument("week", help="the week, YYYY-MM-DD")
    rd.add_argument("--root", default=".")
    rd.set_defaults(fn=_cmd_rederive)
    so = sub.add_parser("store", help="fetch, push and check the content-addressed file store")
    ssub = so.add_subparsers(dest="store_cmd", required=True)
    sf = ssub.add_parser("fetch", help="download a split bundle's files and check their hashes")
    sf.add_argument("bundles", nargs="+", metavar="BUNDLE")
    sf.add_argument("--into", metavar="DIR", help="write a complete copy of each bundle under DIR")
    sf.add_argument("--root", default=".")
    sf.set_defaults(fn=_cmd_store_fetch)
    sp = ssub.add_parser("push", help="verify complete bundles and upload their files")
    sp.add_argument("bundles", nargs="+", metavar="BUNDLE")
    sp.add_argument("--denylist")
    sp.add_argument(
        "--no-username", action="store_true", help="do not scan for the current username"
    )
    sp.add_argument("--root", default=".")
    sp.set_defaults(fn=_cmd_store_push)
    sc = ssub.add_parser("check", help="confirm every listed object is in the store")
    sc.add_argument("bundles", nargs="+", metavar="BUNDLE")
    sc.add_argument("--rehash", type=int, default=0, metavar="N", help="re-download N objects")
    sc.add_argument("--root", default=".")
    sc.set_defaults(fn=_cmd_store_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
