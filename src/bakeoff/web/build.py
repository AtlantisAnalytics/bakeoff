"""Assemble the static site: pages, JSON API, Atom feed and headers.

``render_site`` is pure. It reads ``site.toml``, ``site_src/``, the packaged schema and the week
documents under ``results/``, and returns ``{relative path: bytes}``. Nothing reads the clock or
the environment, so two renders are byte-identical.
"""

import dataclasses
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from xml.sax.saxutils import escape, quoteattr

from bakeoff.publish import week_docs
from bakeoff.web.pages import archive_page, holding_page, week_page, wrap_static
from bakeoff.web.topology import topology_page
from bakeoff.week import load_week_schema
from bakeoff.weekcfg import SiteConfig, load_site

OWNED = (
    "index.html",
    "weeks",
    "api",
    "feed.xml",
    "_headers",
    "method",
    "privacy",
    "topology",
)
ASSETS = (
    "style.css",
    "logo.webp",
    "atlantis.webp",
    "favicon-32.png",
    "apple-touch-icon.png",
    "og.jpg",
)
FEED_EPOCH = "2026-01-01T00:00:00Z"


def _json(obj: object) -> bytes:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def picks_bytes(docs: list[dict]) -> bytes:
    """What ``results/picks.json`` and ``api/v1/picks.json`` contain for these week documents."""
    return _json(docs[-1]["picks_after"] if docs else {})


def https_origin(url: str) -> str:
    """``https://host[:port]`` of an https URL: no userinfo, path or anything else."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme != "https" or "@" in parts.netloc or not re.fullmatch(r"[a-z0-9.-]+", host):
        raise ValueError(f"not an https URL with a plain host: {url!r}")
    return f"https://{host}" + (f":{parts.port}" if parts.port else "")


def headers_file(site: SiteConfig) -> str:
    if site.newsletter_action:
        form_action = https_origin(site.newsletter_action)
    else:
        form_action = "'none'"
    csp = (
        "default-src 'none'; img-src 'self'; style-src 'self'; base-uri 'none'; "
        f"frame-ancestors 'none'; form-action {form_action}"
    )
    return (
        "/*\n"
        f"  Content-Security-Policy: {csp}\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: strict-origin-when-cross-origin\n"
        "\n"
        "/api/v1/*\n"
        "  Access-Control-Allow-Origin: *\n"
    )


def _updated(doc: dict) -> str:
    corrections = doc.get("corrections") or []
    return corrections[-1]["published_at"] if corrections else doc["published_at"]


def _picked_labels(doc: dict) -> str:
    names = {t["id"]: t["name"] for t in doc["tiers"]}
    out = []
    for tier_id in sorted(doc["picks_after"]):
        tracks = doc["picks_after"][tier_id]
        picks = ", ".join(f"{track} {tracks[track]['label']}" for track in sorted(tracks))
        out.append(f"{names.get(tier_id, tier_id)}: {picks}")
    return "; ".join(out)


def atom_feed(docs: list[dict], site: SiteConfig) -> str:
    """One entry per week, newest first. ``docs`` may be in any order."""
    base = site.base_url.rstrip("/")
    ordered = sorted(docs, key=lambda d: d["week"], reverse=True)
    feed_updated = max((_updated(d) for d in ordered), default=FEED_EPOCH)
    out = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<feed xmlns="http://www.w3.org/2005/Atom">',
        "  <title>bakeoff.si</title>",
        "  <id>tag:bakeoff.si,2026:feed</id>",
        f"  <updated>{escape(feed_updated)}</updated>",
        f'  <link rel="self" href={quoteattr(base + "/feed.xml")}/>',
        f'  <link rel="alternate" href={quoteattr(base + "/")}/>',
        "  <author><name>bakeoff.si</name></author>",
    ]
    for d in ordered:
        week = d["week"]
        summary = f"Results for the week of {week}, revision {d['revision']}."
        picks = _picked_labels(d)
        if picks:
            summary += f" Picks: {picks}."
        out += [
            "  <entry>",
            f"    <title>{escape('Week of ' + week)}</title>",
            f"    <id>{escape('tag:bakeoff.si,2026:week/' + week)}</id>",
            f"    <updated>{escape(_updated(d))}</updated>",
            f'    <link rel="alternate" href={quoteattr(base + "/weeks/" + week + "/")}/>',
            f"    <summary>{escape(summary)}</summary>",
            "  </entry>",
        ]
    out += ["</feed>", ""]
    return "\n".join(out)


def _static_page(root: Path, name: str, title: str, site: SiteConfig) -> bytes:
    fragment = (root / "site_src" / f"{name}.html").read_text(encoding="utf-8")
    return wrap_static(fragment, title, site, f"/{name}/").encode("utf-8")


def render_site(root: Path, *, launched: bool | None = None) -> dict[str, bytes]:
    root = Path(root)
    site = load_site(root / "site.toml")
    if launched is not None:
        site = dataclasses.replace(site, launched=launched)
    files: dict[str, bytes] = {}
    docs = week_docs(root) if site.launched else []
    files["index.html"] = (week_page(docs[-1], site) if docs else holding_page(site)).encode(
        "utf-8"
    )
    files["_headers"] = headers_file(site).encode("utf-8")
    files["method/index.html"] = _static_page(root, "method", "bakeoff.si, method", site)
    files["privacy/index.html"] = _static_page(root, "privacy", "bakeoff.si, privacy", site)
    topology = topology_page(root, site)
    if topology is not None:
        files["topology/index.html"] = topology.encode("utf-8")
    if not site.launched:
        return files

    base = site.base_url.rstrip("/")
    newest_first = list(reversed(docs))
    api_index = [
        {
            "week": d["week"],
            "revision": d["revision"],
            "published_at": d["published_at"],
            "url": f"{base}/api/v1/weeks/{d['week']}.json",
        }
        for d in newest_first
    ]
    page_index = [{**e, "url": f"{base}/weeks/{e['week']}/"} for e in api_index]
    files["weeks/index.html"] = archive_page(page_index, site).encode("utf-8")
    for d in docs:
        week = d["week"]
        files[f"weeks/{week}/index.html"] = week_page(d, site).encode("utf-8")
        files[f"api/v1/weeks/{week}.json"] = (root / "results" / week / "week.json").read_bytes()
    if docs:
        files["api/v1/latest.json"] = files[f"api/v1/weeks/{docs[-1]['week']}.json"]
    files["api/v1/weeks/index.json"] = _json(api_index)
    files["api/v1/picks.json"] = picks_bytes(docs)
    files["api/v1/schema/week.v1.json"] = _json(load_week_schema())
    files["feed.xml"] = atom_feed(docs, site).encode("utf-8")
    return files


def _owned_files(out_dir: Path) -> list[str]:
    found = []
    for name in OWNED:
        p = out_dir / name
        if p.is_file() or p.is_symlink():
            found.append(name)
        elif p.is_dir():
            found += [
                q.relative_to(out_dir).as_posix() for q in sorted(p.rglob("*")) if not q.is_dir()
            ]
    return found


def write_site(files: dict[str, bytes], out_dir: Path) -> None:
    """Replace every generator-owned path under ``out_dir`` with ``files``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OWNED:
        p = out_dir / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
    for rel, data in files.items():
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)


def check_site(root: Path) -> list[str]:
    """Problems between ``site/`` (and ``results/picks.json``) and a fresh render."""
    root = Path(root)
    files = render_site(root)
    out_dir = root / "site"
    problems = []
    for rel in sorted(files):
        p = out_dir / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif p.read_bytes() != files[rel]:
            problems.append(f"changed: {rel}")
    problems += [f"stray: {rel}" for rel in _owned_files(out_dir) if rel not in files]
    docs = week_docs(root)
    picks = root / "results" / "picks.json"
    if docs or picks.exists():
        if not picks.is_file():
            problems.append("missing: results/picks.json")
        elif picks.read_bytes() != picks_bytes(docs):
            problems.append("changed: results/picks.json")
    return problems


def write_preview(root: Path, out_dir: Path) -> None:
    """A launched build plus the hand-maintained assets, viewable with ``http.server``."""
    root, out_dir = Path(root), Path(out_dir)
    write_site(render_site(root, launched=True), out_dir)
    for name in ASSETS:
        shutil.copyfile(root / "site" / name, out_dir / name)
