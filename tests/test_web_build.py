import json
import shutil
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest
from htmlcheck import assert_balanced
from weekfix import make_bundle

from bakeoff.publish import publish
from bakeoff.web.build import (
    ASSETS,
    OWNED,
    atom_feed,
    check_site,
    headers_file,
    https_origin,
    render_site,
    write_site,
)
from bakeoff.weekcfg import SiteConfig, load_site

REPO = Path(__file__).resolve().parents[1]
W1, W2 = "2026-10-19", "2026-10-26"
AT1, AT2 = "2026-10-19T07:00:00Z", "2026-10-26T07:00:00Z"
ENV24 = "1× RTX 3090 Ti 24 GB"
GSM = "general/gsm8k"
HOSTILE = 'Evil <b>"x"</b> & co'
ACTION = "https://news.example.org/subscribe?a=1"


def g(k, n=10):
    return [True] * k + [False] * (n - k)


def set_site(root, *, launched, action=""):
    text = (REPO / "site.toml").read_text(encoding="utf-8")
    text = text.replace("launched = false", f"launched = {str(launched).lower()}")
    text = text.replace('action = ""', f'action = "{action}"')
    (root / "site.toml").write_text(text, encoding="utf-8")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    for name in ("tiers.toml", "site.toml"):
        shutil.copy(REPO / name, r / name)
    shutil.copytree(REPO / "site_src", r / "site_src")
    (r / "site").mkdir()
    for name in ("style.css", "logo.webp", "favicon-32.png", "apple-touch-icon.png", "og.jpg"):
        shutil.copy(REPO / "site" / name, r / "site" / name)
    return r


def pub(root, week, at, label, k, *, correction=None):
    b = make_bundle(root / "runs", label=label, weights=label, envelope=ENV24, grades={GSM: g(k)})
    rel = b.relative_to(root).as_posix()
    lines = [f'week = "{week}"', f'published_at = "{at}"', "", "[[run]]", f'bundle = "{rel}"']
    lines += ['tier = "24gb"', 'license = "Apache-2.0"', 'source_url = "https://example.org/m"']
    if correction:
        lines += ["", "[correction]", f'note = "{correction}"']
    toml = root / f"{week}-{at[:4]}.toml"
    toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return publish(toml, root, username="alice", denylist=None)


@pytest.fixture
def two_weeks(root):
    pub(root, W1, AT1, HOSTILE, 6)
    pub(root, W2, AT2, "second", 7)
    set_site(root, launched=True)
    return root


def test_unlaunched_has_only_holding_files(root):
    files = render_site(root)
    assert set(files) == {"index.html", "_headers", "method/index.html", "privacy/index.html"}
    assert b"First results soon" in files["index.html"]


def test_unlaunched_ignores_published_weeks(root):
    pub(root, W1, AT1, "first", 6)
    assert set(render_site(root)) == {
        "index.html",
        "_headers",
        "method/index.html",
        "privacy/index.html",
    }


def test_launched_keys(two_weeks):
    keys = set(render_site(two_weeks))
    for w in (W1, W2):
        assert {f"weeks/{w}/index.html", f"api/v1/weeks/{w}.json"} <= keys
    assert {
        "index.html",
        "weeks/index.html",
        "api/v1/latest.json",
        "api/v1/weeks/index.json",
        "api/v1/picks.json",
        "api/v1/schema/week.v1.json",
        "feed.xml",
        "_headers",
        "method/index.html",
        "privacy/index.html",
    } <= keys


def test_launched_override_without_changing_toml(root):
    pub(root, W1, AT1, "first", 6)
    assert "feed.xml" in render_site(root, launched=True)
    assert "feed.xml" not in render_site(root)


def test_index_is_latest_week(two_weeks):
    files = render_site(two_weeks)
    assert files["index.html"] == files[f"weeks/{W2}/index.html"]
    for name, data in files.items():
        if name.endswith(".html"):
            assert_balanced(data.decode("utf-8"))


def test_api_latest_and_index(two_weeks):
    files = render_site(two_weeks)
    newest = (two_weeks / "results" / W2 / "week.json").read_bytes()
    assert files["api/v1/latest.json"] == newest
    assert files[f"api/v1/weeks/{W2}.json"] == newest
    index = json.loads(files["api/v1/weeks/index.json"])
    assert [e["week"] for e in index] == [W2, W1]
    assert index[0] == {
        "week": W2,
        "revision": 1,
        "published_at": AT2,
        "url": f"https://bakeoff.si/api/v1/weeks/{W2}.json",
    }
    picks = json.loads(files["api/v1/picks.json"])
    assert picks == json.loads(newest)["picks_after"]
    assert files["api/v1/picks.json"] == (two_weeks / "results" / "picks.json").read_bytes()
    assert json.loads(files["api/v1/schema/week.v1.json"])["type"] == "object"


def test_archive_links_to_week_pages(two_weeks):
    html = render_site(two_weeks)["weeks/index.html"].decode("utf-8")
    assert f'href="https://bakeoff.si/weeks/{W2}/"' in html
    assert "/api/v1/" not in html


def test_feed(two_weeks):
    pub(two_weeks, W2, "2026-10-27T08:00:00Z", "second-b", 8, correction="fixed")
    xml = render_site(two_weeks)["feed.xml"].decode("utf-8")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    tree = ET.fromstring(xml)
    entries = tree.findall("a:entry", ns)
    assert [e.find("a:id", ns).text for e in entries] == [
        f"tag:bakeoff.si,2026:week/{W2}",
        f"tag:bakeoff.si,2026:week/{W1}",
    ]
    assert entries[0].find("a:updated", ns).text == "2026-10-27T08:00:00Z"
    assert entries[1].find("a:updated", ns).text == AT1
    assert tree.find("a:updated", ns).text == "2026-10-27T08:00:00Z"
    assert HOSTILE in entries[1].find("a:summary", ns).text
    assert "<b>" not in xml


def test_feed_empty_parses():
    site = SiteConfig(True, "https://bakeoff.si", "https://github.com/x/y", "", "email")
    ET.fromstring(atom_feed([], site))


def test_headers_no_form():
    site = SiteConfig(False, "https://bakeoff.si", "https://github.com/x/y", "", "email")
    h = headers_file(site)
    assert h.startswith("/*\n")
    assert (
        "Content-Security-Policy: default-src 'none'; img-src 'self'; style-src 'self'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'none'\n"
    ) in h
    assert "X-Content-Type-Options: nosniff\n" in h
    assert "Referrer-Policy: strict-origin-when-cross-origin\n" in h
    assert "/api/v1/*\n  Access-Control-Allow-Origin: *\n" in h


def test_headers_form_action_is_origin():
    site = SiteConfig(False, "https://bakeoff.si", "https://github.com/x/y", ACTION, "email")
    assert "form-action https://news.example.org\n" in headers_file(site)


def test_deterministic(two_weeks):
    assert render_site(two_weeks) == render_site(two_weeks)


def test_check_clean_then_changed_missing_stray(two_weeks):
    out = two_weeks / "site"
    write_site(render_site(two_weeks), out)
    assert check_site(two_weeks) == []
    (out / "index.html").write_text("edited", encoding="utf-8")
    assert check_site(two_weeks) == ["changed: index.html"]
    write_site(render_site(two_weeks), out)
    (out / "weeks" / "extra.html").write_text("x", encoding="utf-8")
    (out / "stray-notowned.txt").write_text("x", encoding="utf-8")
    assert check_site(two_weeks) == ["stray: weeks/extra.html"]
    write_site(render_site(two_weeks), out)
    (out / "feed.xml").unlink()
    assert check_site(two_weeks) == ["missing: feed.xml"]


def test_check_reports_stale_picks(two_weeks):
    write_site(render_site(two_weeks), two_weeks / "site")
    (two_weeks / "results" / "picks.json").write_text("{}\n", encoding="utf-8")
    assert check_site(two_weeks) == ["changed: results/picks.json"]


def test_write_site_removes_stale_owned_only(two_weeks):
    out = two_weeks / "site"
    write_site(render_site(two_weeks), out)
    (out / "weeks" / "old.html").write_text("x", encoding="utf-8")
    write_site(render_site(two_weeks, launched=False), out)
    assert not (out / "weeks").exists() and not (out / "api").exists()
    assert not (out / "feed.xml").exists()
    assert (out / "style.css").exists()
    assert set(OWNED) >= {
        p.name
        for p in out.iterdir()
        if p.name
        not in {"style.css", "logo.webp", "favicon-32.png", "apple-touch-icon.png", "og.jpg"}
    }


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://news.example.org/x?y=1", "https://news.example.org"),
        ("https://news.example.org:8443/f", "https://news.example.org:8443"),  # scrub-ok
        ("https://News.Example.org", "https://news.example.org"),
    ],
)
def test_https_origin(url, origin):
    assert https_origin(url) == origin


@pytest.mark.parametrize(
    "url",
    [
        "https://u:tok@news.example.org/x",
        "https://a.example;script-src/x",
        "https:///path",
        "http://news.example.org/",
    ],
)
def test_https_origin_rejects_hostile(url):
    with pytest.raises(ValueError):
        https_origin(url)


class _Refs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ("href", "src") and v:
                self.refs.append(v)


def _assert_links_resolve(files):
    assets = set(ASSETS)
    for name, data in files.items():
        if not name.endswith(".html"):
            continue
        p = _Refs()
        p.feed(data.decode("utf-8"))
        for ref in p.refs:
            if ref.startswith("https://bakeoff.si/"):
                ref = ref[len("https://bakeoff.si") :]
            if not ref.startswith("/"):
                assert ref.startswith("https://"), (name, ref)
                continue
            path = ref.split("#")[0].lstrip("/")
            if path.endswith("/") or path == "":
                path += "index.html"
            assert path in files or path in assets, (name, ref)


def test_links_resolve_unlaunched(root):
    _assert_links_resolve(render_site(root))


def test_links_resolve_launched(two_weeks):
    files = render_site(two_weeks)
    _assert_links_resolve(files)
    # same-origin links in pages that are not .html (API/feed) are not checked here
    assert "weeks/index.html" in files


def test_week_page_never_links_javascript(two_weeks):
    from bakeoff.publish import week_docs
    from bakeoff.web.pages import week_page

    doc = week_docs(two_weeks)[-1]
    model = next(r for t in doc["tiers"] for r in t["runs"])["model"]
    model["source_url"] = "javascript:alert(1)"
    model["license_url"] = "data:text/html,x"
    html = week_page(doc, load_site(two_weeks / "site.toml"))
    assert 'href="javascript:' not in html and 'href="data:' not in html
    assert "Source" in html
