import shutil
from pathlib import Path

import pytest
from htmlcheck import assert_balanced
from topofix import RUNS, bundle
from topofix import write_experiment as setup

from bakeoff.publish import PublishError
from bakeoff.web.build import render_site
from bakeoff.web.topology import topology_page
from bakeoff.weekcfg import SiteConfig

REPO = Path(__file__).resolve().parents[1]
SITE = SiteConfig(True, "https://bakeoff.si", "https://github.com/o/r", "", "email")


def cells(html):
    rows = html.split("<tr>")[2:]  # after the header row
    return [[c.split(">", 1)[1] for c in r.split("</td>")[:-1]] for r in rows]


def test_no_experiment_no_page(tmp_path):
    assert topology_page(tmp_path, SITE) is None


def test_table_from_bundles(tmp_path):
    one = bundle(
        tmp_path,
        f"{RUNS}/a",
        envelope="1× RTX 3060 12 GB",
        parallel="",
        per_gpu=(9500,),
        paths=(),
        dec=46.31,
        agg=158.2,
        ttft=0.042,
        power=136.0,
    )
    two = bundle(tmp_path, f"{RUNS}/b", paths=("SYS",))
    setup(tmp_path, {"g3": [one, two]}, "<h2>Granite</h2>\n<!-- runs: g3 -->\n<p>After.</p>")
    h = topology_page(tmp_path, SITE)
    assert_balanced(h)
    assert "<h2>Granite</h2>" in h and "<p>After.</p>" in h and "<!-- runs" not in h
    assert cells(h) == [
        ["1× RTX 3060 12 GB", "one card", "1", "—", "46.3", "158.2", "42 ms", "32K", "136 W",
         '<a href="https://github.com/o/r/tree/main/results/experiments/2026-11-multi-gpu/runs/a">'
         "Raw bundle</a>"],
        ["2× RTX 3060 12 GB", "tensor parallel", "2 of 2", "across sockets", "79.1", "280.1",
         "33 ms", "32K", "245 W",
         '<a href="https://github.com/o/r/tree/main/results/experiments/2026-11-multi-gpu/runs/b">'
         "Raw bundle</a>"],
    ]  # fmt: skip


def test_cards_used_counts_cards_holding_a_gigabyte(tmp_path):
    # a layer split can leave the last card empty: the bundle's memory peaks show it
    r = bundle(tmp_path, f"{RUNS}/s", parallel="layer-split", per_gpu=(11000, 10500, 4),
               paths=("SYS", "SYS", "NODE"))  # fmt: skip
    setup(tmp_path, {"q": [r]}, "<!-- runs: q -->")
    row = cells(topology_page(tmp_path, SITE))[0]
    assert row[1:4] == ["layer split", "2 of 3", "across sockets"]


@pytest.mark.parametrize(
    ("parallel", "paths", "split", "link"),
    [
        ("pp=3", ("NODE", "NODE"), "pipeline", "same socket"),
        ("tp=2", ("PIX",), "tensor parallel", "same PCIe switch"),
        ("tp=2", ("NV4",), "tensor parallel", "NVLink"),
        ("ep=2", ("WEIRD",), "ep=2", "WEIRD"),
    ],
)
def test_split_and_link_labels(tmp_path, parallel, paths, split, link):
    r = bundle(tmp_path, f"{RUNS}/x", parallel=parallel, paths=paths)
    setup(tmp_path, {"g": [r]}, "<!-- runs: g -->")
    row = cells(topology_page(tmp_path, SITE))[0]
    assert (row[1], row[3]) == (split, link)


def test_published_week_rows_say_so(tmp_path):
    r = bundle(tmp_path, "results/2026-11-02/runs/w", envelope="1× RTX 3090 Ti 24 GB",
               parallel="", per_gpu=(20000,), paths=())  # fmt: skip
    setup(tmp_path, {"g": [r]}, "<!-- runs: g -->")
    h = topology_page(tmp_path, SITE)
    week = '<br><span class="muted small">week 2026-11-02</span>'
    assert cells(h)[0][0] == "1× RTX 3090 Ti 24 GB" + week


def test_values_are_escaped(tmp_path):
    r = bundle(tmp_path, f"{RUNS}/e", envelope="<b>x</b>", parallel="<i>")
    setup(tmp_path, {"g": [r]}, "<!-- runs: g -->")
    h = topology_page(tmp_path, SITE)
    assert "<b>" not in h and "<i>" not in h and "&lt;b&gt;x&lt;/b&gt;" in h


@pytest.mark.parametrize(
    ("groups", "page", "msg"),
    [
        ({"g": []}, "<!-- runs: h -->", "no group 'h'"),
        ({"g": [], "h": []}, "<!-- runs: g -->", "not on the page: h"),
        ({"g": ["results/../etc/x"]}, "<!-- runs: g -->", "bundle path"),
        ({"g": ["site/x"]}, "<!-- runs: g -->", "bundle path"),
        ({"g": [f"{RUNS}/missing"]}, "<!-- runs: g -->", "cannot read"),
    ],
)
def test_manifest_and_page_must_agree(tmp_path, groups, page, msg):
    setup(tmp_path, groups, page)
    with pytest.raises((PublishError, ValueError), match=msg):
        topology_page(tmp_path, SITE)


def test_incomplete_bundle_refused(tmp_path):
    r = bundle(tmp_path, f"{RUNS}/i")
    p = tmp_path / r / "bundle.json"
    p.write_text(p.read_text().replace('"complete"', '"incomplete"'), encoding="utf-8")
    setup(tmp_path, {"g": [r]}, "<!-- runs: g -->")
    with pytest.raises(ValueError, match="not complete"):
        topology_page(tmp_path, SITE)


def test_render_site_includes_the_page(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(REPO / "site_src", root / "site_src")
    shutil.copy(REPO / "site.toml", root / "site.toml")
    r = bundle(root, f"{RUNS}/a")
    setup(root, {"g": [r]}, "<!-- runs: g -->")
    files = render_site(root, launched=False)
    assert "topology/index.html" in files
    assert b"<title>bakeoff.si, multi-GPU topology</title>" in files["topology/index.html"]


def test_repo_page_matches_its_bundles():
    """The real page: every group placed, and the headline numbers come from the bundles."""
    h = topology_page(REPO, SITE)
    assert h is not None
    assert_balanced(h)
    for bad in ("<style", "style=", "<script"):
        assert bad not in h
    # one table per group, in experiment.toml order: granite-3b, granite-8b, qwen-27b, gemma-12b
    tables = [cells(t.split("</table>")[0]) for t in h.split("<table>")[1:]]
    granite, _, qwen, _ = ({(r[0], r[1], r[3]): r for r in t} for t in tables)
    assert granite[("2× RTX 3060 12 GB", "tensor parallel", "across sockets")][4] == "78.7"
    assert qwen[("2× RTX 3060 12 GB", "tensor parallel", "across sockets")][4] == "32.6"
    assert qwen[("3× RTX 3060 12 GB", "layer split", "across sockets")][2] == "2 of 3"
