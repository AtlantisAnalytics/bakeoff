"""The multi-GPU topology page: prose from ``site_src/topology.html``, tables from bundles.

Each ``<!-- runs: <key> -->`` line in the fragment becomes a table of that group's runs, as listed
in the experiment's ``experiment.toml``. Every number in a table is read from a bundle, so
``bakeoff site --check`` covers them as it covers the weeks.
"""

import re
import tomllib
from pathlib import Path

from bakeoff.publish import _read_json
from bakeoff.web import fmt
from bakeoff.web.pages import _e, _link, wrap_static
from bakeoff.weekcfg import SiteConfig

EXPERIMENT = "results/experiments/2026-11-multi-gpu"
TITLE = "bakeoff.si, multi-GPU topology"

_MARKER = re.compile(r"<!-- runs: ([a-z0-9-]+) -->")
_BUNDLE = re.compile(r"results/[A-Za-z0-9][A-Za-z0-9._/-]*")
_WEEK = re.compile(r"results/(\d{4}-\d{2}-\d{2})/runs/")
_USED_MIB = 1024  # a card holding less than this during the run is not carrying the model
# nvidia-smi topo -m paths, nearest first
_PATHS = {
    "PIX": "same PCIe switch",
    "PXB": "PCIe switches",
    "PHB": "same host bridge",
    "NODE": "same socket",
    "SYS": "across sockets",
}
_ORDER = ["NV", *_PATHS]
_SPLITS = {"tp": "tensor parallel", "pp": "pipeline"}
_HEAD_ROW = (
    "<tr><th>Cards</th><th>Split</th><th>Cards used</th><th>Farthest link</th>"
    '<th class="num">Decode tok/s</th><th class="num">4 streams tok/s</th>'
    '<th class="num">TTFT</th><th class="num">Context</th><th class="num">Power</th>'
    "<th>Data</th></tr>"
)


def _value(leaf: object) -> object:
    return leaf.get("value") if isinstance(leaf, dict) else None


def _median(speed: dict, cell: str, metric: str) -> float | None:
    v = (speed.get(cell) or {}).get(metric)
    return v.get("median") if isinstance(v, dict) else None


def _split(parallel: str) -> str:
    if not parallel:
        return "one card"
    if parallel == "layer-split":
        return "layer split"
    return _SPLITS.get(parallel.split("=")[0], parallel)


def _farthest(paths: list) -> str:
    known = [p for p in paths if isinstance(p, str)]
    if not known:
        return fmt.DASH

    def rank(p: str) -> int:
        return next((i for i, k in enumerate(_ORDER) if p.startswith(k)), len(_ORDER))

    far = max(known, key=rank)
    return "NVLink" if far.startswith("NV") else _PATHS.get(far, far)


def _cards_used(meta: dict, n: int) -> str:
    peaks = _value(meta.get("memory", {}).get("peaks")) or {}
    used = set()
    for p in peaks.values():
        for i, mib in enumerate(p.get("mem_used_mib_per_gpu") or []):
            if isinstance(mib, (int, float)) and mib >= _USED_MIB:
                used.add(i)
    return str(n) if n == 1 else f"{len(used)} of {n}"


def _row(root: Path, rel: str, site: SiteConfig) -> str:
    if not _BUNDLE.fullmatch(rel) or ".." in rel.split("/"):
        raise ValueError(f"{EXPERIMENT}/experiment.toml: bad bundle path {rel!r}")
    meta = _read_json(root / rel / "bundle.json", f"{rel}/bundle.json")
    if not isinstance(meta, dict) or meta.get("status") != "complete":
        raise ValueError(f"{rel}: run is not complete")
    model = meta["config"]["model"]
    env = meta.get("env", {})
    speed = meta.get("speed") or {}
    power = (_value(meta.get("memory", {}).get("peaks")) or {}).get("speed", {}).get("power_w_mean")
    cards = _e(model["envelope"])
    week = _WEEK.match(rel)
    if week:
        cards += f'<br><span class="muted small">week {_e(week.group(1))}</span>'
    href = f"{site.repo_url.rstrip('/')}/tree/main/{rel}"
    cols = [
        f"<td>{cards}</td>",
        f"<td>{_e(_split(model.get('parallel', '')))}</td>",
        f"<td>{_e(_cards_used(meta, len(env.get('gpus', []))))}</td>",
        f"<td>{_e(_farthest([_value(lk.get('path')) for lk in env.get('gpu_links', [])]))}</td>",
        f'<td class="num">{fmt.tps(_median(speed, "short@1", "decode_tps"))}</td>',
        f'<td class="num">{fmt.tps(_median(speed, "short@4", "aggregate_tps"))}</td>',
        f'<td class="num">{fmt.ms(_median(speed, "short@1", "ttft_s"))}</td>',
        f'<td class="num">{fmt.kctx((meta.get("fit") or {}).get("max_context_passed"))}</td>',
        f'<td class="num">{fmt.DASH if power is None else f"{power:.0f} W"}</td>',
        f"<td>{_link(href, 'Raw bundle')}</td>",
    ]
    return "<tr>" + "".join(cols) + "</tr>"


def topology_page(root: Path, site: SiteConfig) -> str | None:
    """The page, or None when the experiment is not in this tree."""
    root = Path(root)
    manifest = root / EXPERIMENT / "experiment.toml"
    if not manifest.is_file():
        return None
    groups = {g["key"]: g["bundles"] for g in tomllib.loads(manifest.read_text("utf-8"))["group"]}
    fragment = (root / "site_src" / "topology.html").read_text(encoding="utf-8")
    placed = _MARKER.findall(fragment)
    if unknown := [k for k in placed if k not in groups]:
        raise ValueError(f"site_src/topology.html: no group {unknown[0]!r} in experiment.toml")
    if missing := sorted(set(groups) - set(placed)):
        raise ValueError(f"site_src/topology.html: groups not on the page: {', '.join(missing)}")

    def table(m: re.Match) -> str:
        key = m.group(1)
        rows = "".join(_row(root, rel, site) for rel in groups[key])
        return f'<div class="table-wrap"><table>{_HEAD_ROW}{rows}</table></div>'

    return wrap_static(_MARKER.sub(table, fragment), TITLE, site, "/topology/")
