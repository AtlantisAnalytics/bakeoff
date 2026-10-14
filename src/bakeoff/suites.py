"""Suite loading, hashing, resolution and lock checking."""

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bakeoff
from bakeoff.hashing import canonical_json, sha256_hex
from bakeoff.needle import needle_messages

SPLITS = ("public", "private")


class SuiteError(Exception):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    messages: list[dict]
    expected: Any


@dataclass
class Suite:
    path: Path
    track: str
    name: str
    version: int
    split: str
    license: str
    source: dict
    grader: str
    grader_options: dict
    max_tokens: int
    system: str
    items: list[Item] = field(default_factory=list)
    generator: dict | None = None
    hash: str = ""

    @property
    def key(self) -> str:
        return f"{self.track}/{self.name}"


def _require_str(m: dict, k: str, where: Path) -> str:
    v = m.get(k)
    if not isinstance(v, str) or not v:
        raise SuiteError(f"{where}: manifest field {k!r} must be a non-empty string")
    return v


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _generate(gen: dict, system: str, where: Path) -> list[Item]:
    if gen.get("name") != "needle":
        raise SuiteError(f"{where}: unknown generator {gen.get('name')!r}")
    try:
        lines, depths, seed = gen["lines"], gen["depths"], gen["seed"]
    except KeyError as e:
        raise SuiteError(f"{where}: generator is missing {e.args[0]!r}") from e
    items = []
    for n in lines:
        for depth in depths:
            msgs, key = needle_messages(n, depth, seed)
            if system:
                msgs = [{"role": "system", "content": system}, *msgs]
            items.append(Item(f"needle-{n}-{int(depth * 100)}", msgs, key))
    return items


def _read_items(path: Path, system: str) -> tuple[list[dict], list[Item]]:
    raw: list[dict] = []
    items: list[Item] = []
    f = path / "items.jsonl"
    if not f.is_file():
        raise SuiteError(f"{path}: missing items.jsonl")
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise SuiteError(f"{f}:{n}: invalid JSON: {e}") from e
        if not isinstance(d, dict) or "id" not in d or "expected" not in d:
            raise SuiteError(f"{f}:{n}: item needs 'id' and 'expected'")
        if "messages" in d:
            msgs = list(d["messages"])
        elif "prompt" in d:
            msgs = [{"role": "user", "content": d["prompt"]}]
        else:
            raise SuiteError(f"{f}:{n}: item needs 'messages' or 'prompt'")
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        raw.append(d)
        items.append(Item(str(d["id"]), msgs, d["expected"]))
    return raw, items


def load_suite(path: Path) -> Suite:
    path = Path(path)
    mf = path / "manifest.toml"
    if not mf.is_file():
        raise SuiteError(f"{path}: missing manifest.toml")
    try:
        manifest = tomllib.loads(mf.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise SuiteError(f"{mf}: invalid TOML: {e}") from e

    track = _require_str(manifest, "track", mf)
    name = _require_str(manifest, "name", mf)
    grader = _require_str(manifest, "grader", mf)
    split = manifest.get("split")
    if split not in SPLITS:
        raise SuiteError(f"{mf}: split must be one of {SPLITS}, got {split!r}")
    version = manifest.get("version")
    if not _is_int(version) or version < 1:
        raise SuiteError(f"{mf}: version must be an integer >= 1")
    max_tokens = manifest.get("max_tokens")
    if not _is_int(max_tokens) or max_tokens <= 0:
        raise SuiteError(f"{mf}: max_tokens must be an integer > 0")
    system = manifest.get("system", "")
    if not isinstance(system, str):
        raise SuiteError(f"{mf}: system must be a string")

    gen = manifest.get("generator")
    if gen is not None:
        if not isinstance(gen, dict):
            raise SuiteError(f"{mf}: generator must be a table")
        raw_items: list[dict] = []
        items = _generate(gen, system, mf)
    else:
        raw_items, items = _read_items(path, system)

    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise SuiteError(f"{path}: duplicate item ids")

    return Suite(
        path=path,
        track=track,
        name=name,
        version=version,
        split=split,
        license=str(manifest.get("license", "")),
        source=dict(manifest.get("source", {})),
        grader=grader,
        grader_options=dict(manifest.get("grader_options", {})),
        max_tokens=max_tokens,
        system=system,
        items=items,
        generator=dict(gen) if gen is not None else None,
        hash=sha256_hex(canonical_json({"manifest": manifest, "items": raw_items})),
    )


def suites_root() -> Path:
    here = Path(bakeoff.__file__).resolve().parent
    if (here / "suites").is_dir():
        return here / "suites"
    for parent in here.parents:
        if (parent / "suites").is_dir() and (parent / "pyproject.toml").is_file():
            return parent / "suites"
    return here.parents[1] / "suites"


def _find(directory: Path) -> list[Path]:
    if (directory / "manifest.toml").is_file():
        return [directory]
    return sorted(m.parent for m in directory.rglob("manifest.toml"))


def resolve_suites(refs: list[str]) -> list[Suite]:
    found: dict[str, Suite] = {}
    for ref in refs:
        p = Path(ref)
        if not p.is_dir():
            p = suites_root() / ref
        if not p.is_dir():
            raise SuiteError(f"suite not found: {ref}")
        dirs = _find(p)
        if not dirs:
            raise SuiteError(f"no suites under {ref}")
        for d in dirs:
            s = load_suite(d)
            if s.key in found and found[s.key].path.resolve() != d.resolve():
                raise SuiteError(f"duplicate suite key {s.key}")
            found[s.key] = s
    return [found[k] for k in sorted(found)]


def check_lock(found: list[Suite], lock: dict) -> list[str]:
    problems = []
    for s in found:
        versions = lock.get(s.key, {})
        v = str(s.version)
        if v not in versions:
            problems.append(f"{s.key} version {v} is not in the lock: run scripts/lock_suites.py")
        elif versions[v] != s.hash:
            problems.append(
                f"{s.key} version {v} changed since it was locked: bump `version` "
                "(then run scripts/lock_suites.py)"
            )
    return problems
