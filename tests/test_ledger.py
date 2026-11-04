from pathlib import Path

import pytest

from bakeoff.ledger import LedgerError, bundle_kind, load_layout, load_withdrawn, week_of

LAYOUT = """
[[layout]]
from = "2026-11-02"
keep = "all"

[[layout]]
from = "2026-11-30"
run = ["bundle.json", "speed.jsonl", "fit.jsonl"]
judge = ["bundle.json"]
"""

SHA = "a" * 64


def write(root: Path, text: str, name: str = "layout.toml") -> None:
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / name).write_text(text)


def test_ranges_select_by_week(tmp_path):
    write(tmp_path, LAYOUT)
    lay = load_layout(tmp_path)
    assert lay.kept("2026-11-23", "run") is None
    assert lay.kept("2026-11-30", "run") == {"bundle.json", "speed.jsonl", "fit.jsonl"}
    assert lay.kept("2027-01-04", "judge") == {"bundle.json"}


@pytest.mark.parametrize(
    "text, match",
    [
        (
            '[[layout]]\nfrom = "2026-11-30"\nkeep = "all"\n[[layout]]\nfrom = "2026-11-02"\n'
            'keep = "all"\n',
            "sorted",
        ),
        (
            '[[layout]]\nfrom = "2026-11-02"\nkeep = "all"\n[[layout]]\nfrom = "2026-11-02"\n'
            'keep = "all"\n',
            "repeat",
        ),
        ('[[layout]]\nfrom = "2026-11-02"\nkeep = "some"\n', "keep"),
        ('[[layout]]\nfrom = "Nov 2"\nkeep = "all"\n', "from"),
        ('[[layout]]\nfrom = "2026-11-02"\n', "run"),
        ("", "layout"),
    ],
)
def test_layout_errors(tmp_path, text, match):
    write(tmp_path, text)
    with pytest.raises(LedgerError, match=match):
        load_layout(tmp_path)


def test_layout_lookup_errors(tmp_path):
    write(tmp_path, LAYOUT)
    lay = load_layout(tmp_path)
    with pytest.raises(LedgerError, match="2026-10-26"):
        lay.kept("2026-10-26", "run")
    with pytest.raises(LedgerError, match="community"):
        lay.kept("2026-11-30", "community")


def test_missing_layout_keeps_everything(tmp_path):
    assert load_layout(tmp_path).kept("2099-01-05", "run") is None


def row(sha=SHA, date="2027-01-12", reason="why", bundles='["results/2026-12-07/runs/x"]'):
    return (
        f'[[withdrawn]]\nsha256 = "{sha}"\ndate = "{date}"\nreason = "{reason}"\n'
        f"bundles = {bundles}\n"
    )


def test_withdrawn_rows(tmp_path):
    assert load_withdrawn(tmp_path) == {}
    write(tmp_path, row(), "withdrawn.toml")
    w = load_withdrawn(tmp_path)[SHA]
    assert (w.date, w.reason, w.bundles) == ("2027-01-12", "why", ("results/2026-12-07/runs/x",))


@pytest.mark.parametrize(
    "text, match",
    [
        (row(sha="A" * 64), "sha256"),
        (row(sha="a" * 63), "sha256"),
        (row(date="2027-13-01"), "date"),
        (row(date="12/01/2027"), "date"),
        (row(reason=""), "reason"),
        (row(bundles="[]"), "bundles"),
        (row() + row(), "twice"),
    ],
)
def test_withdrawn_errors(tmp_path, text, match):
    write(tmp_path, text, "withdrawn.toml")
    with pytest.raises(LedgerError, match=match):
        load_withdrawn(tmp_path)


def test_week_of_and_kind(tmp_path):
    assert week_of(tmp_path, tmp_path / "results/2026-11-30/runs/x") == "2026-11-30"
    assert week_of(tmp_path, tmp_path / "results/2026-11-30/judge/j") == "2026-11-30"
    assert week_of(tmp_path, tmp_path / "results/experiments/2026-11-multi-gpu/runs/x") is None
    assert week_of(tmp_path, tmp_path / "elsewhere/2026-11-30/runs/x") is None
    assert bundle_kind({"bundle_format": "judge-m1"}) == "judge"
    assert bundle_kind({"bundle_format": "m2"}) == "run"
