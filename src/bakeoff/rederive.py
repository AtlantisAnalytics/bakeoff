"""``bakeoff rederive <week>``: re-publish a published week from the repository and compare.

Everything comes from the checkout, so anyone with a clone can check a week:

* ``results/<week>/week.toml``, the input exactly as it was published;
* the published bundle copies: each ``[[run]]`` maps to ``results/<week>/runs/<dir>`` and each
  ``[[judge]]`` to ``results/<week>/judge/<dir>``, where ``<dir>`` is the last segment of the
  ``bundle`` path in ``week.toml``;
* the ``week.json`` of every earlier week, so ``picks_before`` derives exactly as it did;
* ``tiers.toml``, limited to the tiers the stored week lists, in ``tiers.toml`` order. A tier added
  after the week was published is not part of it; a stored tier missing from ``tiers.toml`` is an
  error. (Dropping a tier that has runs or carried picks from the stored week is still detected:
  the re-publish then refuses.)

The work happens in a temporary root: the bundles are copied there, a ``week.toml`` with only the
``bundle`` paths remapped is written (every field that reaches ``week.json`` is kept as is), and
:func:`bakeoff.publish.publish` runs on it. The new ``week.json`` is compared byte for byte with
``results/<week>/week.json``. Nothing under the real ``results/`` is written.

Corrections (``revision`` > 1). The stored ``week.toml`` is the latest correction's input, so the
re-publish runs as that correction on top of the previous revision. Earlier revisions' inputs live
only in git history, so the previous revision's bookkeeping is seeded from the stored document:
``revision - 1``, the original ``published_at`` and every correction entry but the last. Everything
else, including the last correction entry (from ``[correction]`` and ``published_at``), is
re-derived and compared.

Split weeks. For a week that ``results/layout.toml`` puts in a split range, the published bundle
directories hold only the kept files. Each bundle is rebuilt in full from the store (verified by
sha256 and cached), ``layout.toml`` and ``withdrawn.toml`` are copied beside it, and the
re-publish runs without the existence check: it reproduces the verdicts, it never publishes. A
withdrawn file outside ``quality/`` is left out and tolerated; a withdrawn quality file means the
week can no longer be re-derived. A second run works offline from a warm cache.
"""

import json
import re
import shutil
import tempfile
import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath

from bakeoff.ledger import LedgerError, load_layout
from bakeoff.publish import PublishError, publish
from bakeoff.store import Store, StoreError, materialize
from bakeoff.weekcfg import WeekInputError, load_tiers

_WEEK = re.compile(r"\d{4}-\d{2}-\d{2}")
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHOWN = 10


class RederiveError(Exception):
    pass


def _q(v: object, where: str) -> str:
    """A TOML basic string. JSON string escapes are a subset of TOML's; JSON leaves DEL raw, which
    TOML forbids."""
    if not isinstance(v, str):
        raise RederiveError(f"week.toml {where} is not a string")
    return json.dumps(v, ensure_ascii=False).replace("\x7f", "\\u007f")


def _dumps(obj: object) -> bytes:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _dir_name(bundle: object, where: str) -> str:
    if not isinstance(bundle, str):
        raise RederiveError(f"week.toml {where}.bundle is not a string")
    name = PureWindowsPath(bundle).name if "\\" in bundle else PurePosixPath(bundle).name
    if not _SEGMENT.fullmatch(name):
        raise RederiveError(f"week.toml {where}.bundle does not end in a safe directory name")
    return name


def _table(lines: list[str], header: str, table: object, where: str) -> None:
    if not isinstance(table, dict):
        raise RederiveError(f"week.toml {where} is not a table")
    lines += ["", header]
    lines += [f"{k} = {_q(table[k], f'{where}.{k}')}" for k in sorted(table)]


def _copy_bundle(src: Path, dst: Path, what: str, store: Store | None = None) -> None:
    if src.is_symlink() or not src.is_dir():
        raise RederiveError(f"{what}: published bundle not found: {src.name}")
    if store is None:
        shutil.copytree(src, dst, symlinks=True)
        return
    try:
        meta = json.loads((src / "bundle.json").read_bytes().decode("utf-8"))
        files = meta["files"]
        for rel in sorted(files):
            w = store.withdrawn.get(files[rel])
            if w is not None and rel.startswith("quality/"):
                raise RederiveError(
                    f"{rel} in {src.name} was withdrawn on {w.date}: {w.reason}; "
                    "this week can no longer be re-derived"
                )
        materialize(src, dst, store, skip_withdrawn=True)
    except (OSError, ValueError, KeyError, TypeError, StoreError) as e:
        raise RederiveError(f"{what}: cannot rebuild {src.name} from the store: {e}") from e


def _remapped_week_toml(data: dict, week_dir: Path, tmp: Path, store: Store | None) -> str:
    """Copy the published bundles into ``tmp`` and return week.toml text pointing at them."""
    lines = [f"week = {_q(data.get('week'), 'week')}"]
    lines.append(f"published_at = {_q(data.get('published_at'), 'published_at')}")
    for kind, sub in (("run", "runs"), ("judge", "judge")):
        tables = data.get(kind, [])
        if not isinstance(tables, list):
            raise RederiveError(f"week.toml {kind} is not an array of tables")
        for i, t in enumerate(tables):
            where = f"{kind}[{i}]"
            if not isinstance(t, dict):
                raise RederiveError(f"week.toml {where} is not a table")
            name = _dir_name(t.get("bundle"), where)
            if (tmp / sub / name).exists():
                raise RederiveError(f"week.toml {where}: directory name {name} is repeated")
            _copy_bundle(week_dir / sub / name, tmp / sub / name, where, store)
            _table(lines, f"[[{kind}]]", {**t, "bundle": f"{sub}/{name}"}, where)
    for i, t in enumerate(data.get("override", [])):
        _table(lines, "[[override]]", t, f"override[{i}]")
    if "correction" in data:
        _table(lines, "[correction]", data["correction"], "correction")
    return "\n".join(lines) + "\n"


def _tiers_toml(week_dir: Path, stored: dict) -> str:
    """The tiers the week was published with: its own ``tiers.toml`` copy, never today's file (a
    tier renamed, reordered, removed or launched since must not change an old week). A week
    published before that copy existed takes them from its ``week.json``."""
    if (week_dir / "tiers.toml").is_file():
        text = (week_dir / "tiers.toml").read_text(encoding="utf-8")
        try:
            load_tiers(week_dir / "tiers.toml")
        except WeekInputError as e:
            raise RederiveError(f"tiers.toml: {e}") from e
        return text
    lines = []
    for i, t in enumerate(stored.get("tiers", [])):
        if not isinstance(t, dict):
            raise RederiveError(f"the stored week.json tiers[{i}] is not an object")
        lines += [
            "[[tier]]",
            f"id = {_q(t.get('id'), f'tiers[{i}].id')}",
            f"name = {_q(t.get('name'), f'tiers[{i}].name')}",
            f"envelope = {_q(t.get('envelope'), f'tiers[{i}].envelope')}",
            *(["coming = true"] if t.get("coming") is True else []),
            "",
        ]
    return "\n".join(lines)


def _previous_revision(stored: dict) -> dict:
    """The stored document as it stood one revision earlier (bookkeeping only)."""
    rev, corrections = stored.get("revision"), stored.get("corrections")
    if not isinstance(rev, int) or isinstance(rev, bool) or not isinstance(corrections, list):
        raise RederiveError("the stored week.json has no valid revision or corrections")
    return {**stored, "revision": rev - 1, "corrections": corrections[:-1]}


def _diff(a: object, b: object, path: str, out: list[str]) -> None:
    if len(out) > _SHOWN:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only in the stored week.json")
            elif k not in b:
                out.append(f"{path}.{k}: missing from the stored week.json")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff(x, y, f"{path}[{i}]", out)
    elif a != b or type(a) is not type(b):
        out.append(f"{path}: re-derived {json.dumps(a)[:60]}, stored {json.dumps(b)[:60]}")


def _summary(produced: bytes, stored: bytes) -> list[str]:
    try:
        a, b = json.loads(produced), json.loads(stored)
    except ValueError:
        return ["week.json: the stored file is not valid JSON"]
    out: list[str] = []
    _diff(a, b, "week", out)
    if not out:
        return ["week.json: same content, different bytes (formatting)"]
    if len(out) > _SHOWN:
        out = [*out[:_SHOWN], "and more"]
    return out


def rederive(root: Path, week: str, store: Store | None = None) -> list[str]:
    """Re-publish ``week`` in a temporary root; return the differences (empty when identical)."""
    root = Path(root)
    if not _WEEK.fullmatch(week):
        raise RederiveError(f"week must be YYYY-MM-DD, got {week!r}")
    results = root / "results"
    week_dir = results / week
    if not (week_dir / "week.toml").is_file() or not (week_dir / "week.json").is_file():
        raise RederiveError(f"results/{week} has no published week.toml and week.json")
    stored_bytes = (week_dir / "week.json").read_bytes()
    try:
        source = (week_dir / "week.toml").read_bytes().decode("utf-8")
        data = tomllib.loads(source)
        stored = json.loads(stored_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as e:
        raise RederiveError(f"cannot read results/{week}: {e}") from e
    if not isinstance(stored, dict):
        raise RederiveError(f"results/{week}/week.json is not an object")

    try:
        if load_layout(root).kept(week, "run") is None:
            store = None  # a keep = "all" week needs no store
        elif store is None:
            store = Store.from_env(root)
    except (LedgerError, StoreError) as e:
        raise RederiveError(f"layout: {e}") from e

    with tempfile.TemporaryDirectory(prefix="bakeoff-rederive-") as t:
        tmp = Path(t)
        for name in ("layout.toml", "withdrawn.toml"):
            if (results / name).is_file():
                (tmp / "results").mkdir(exist_ok=True)
                shutil.copyfile(results / name, tmp / "results" / name)
        (tmp / "tiers.toml").write_text(_tiers_toml(week_dir, stored), encoding="utf-8")
        if (root / "site.toml").is_file():
            shutil.copyfile(root / "site.toml", tmp / "site.toml")
        for d in sorted(results.iterdir()) if results.is_dir() else []:
            if _WEEK.fullmatch(d.name) and d.name < week:
                (tmp / "results" / d.name).mkdir(parents=True)
                shutil.copyfile(d / "week.json", tmp / "results" / d.name / "week.json")
        if "correction" in data:
            (tmp / "results" / week).mkdir(parents=True)
            (tmp / "results" / week / "week.json").write_bytes(_dumps(_previous_revision(stored)))
        toml = tmp / "week.toml"
        toml.write_text(_remapped_week_toml(data, week_dir, tmp, store), encoding="utf-8")
        try:
            out = publish(toml, tmp, username=None, denylist=None, store=store, check_store=False)
        except PublishError as e:
            raise RederiveError(f"re-publish failed: {e}") from e
        produced = (out / "week.json").read_bytes()
    return [] if produced == stored_bytes else _summary(produced, stored_bytes)
