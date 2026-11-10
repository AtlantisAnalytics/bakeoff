import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from weekfix import KEPT_RUN, make_bundle, push_all, split

from bakeoff import cli
from bakeoff.ledger import LedgerError
from bakeoff.ledgercheck import changed_bundles, check_ledger
from bakeoff.store import object_key

REPO = Path(__file__).resolve().parents[1]
SPLIT_LAYOUT = (
    '[[layout]]\nfrom = "2026-11-02"\nkeep = "all"\n[[layout]]\nfrom = "2026-11-30"\n'
    'run = ["bundle.json", "speed.jsonl", "fit.jsonl"]\njudge = ["bundle.json"]\n'
)


def _split_root(tmp_path, week="2026-11-30"):
    root = tmp_path / "repo"
    run = make_bundle(
        root / "results" / week / "runs",
        label="A",
        weights="w",
        envelope="e",
        grades={"general/x": [True, False]},
    )
    (root / "results" / "layout.toml").write_text(SPLIT_LAYOUT)
    return root, run


def _kinds(found):
    return sorted((b, f.file, f.kind) for b, f in found)


def test_ledger_clean_repo():
    assert check_ledger(REPO, username=None, denylist=None) == []


def test_ledger_split_bundle_clean(tmp_path):
    root, run = _split_root(tmp_path)
    split(run, KEPT_RUN | {"bundle.json"})
    assert check_ledger(root, username=None, denylist=None) == []


def _status_root(tmp_path, status):
    root = tmp_path / "repo"
    run = make_bundle(
        root / "results" / "2026-11-09" / "runs",
        label="A",
        weights="w",
        envelope="e",
        grades={"general/x": [True]},
        status=status,
    )
    (root / "results" / "layout.toml").write_text(SPLIT_LAYOUT)
    return root, run


def test_ledger_accepts_incomplete_bundle(tmp_path):
    # publish puts incomplete bundles in the week, so the ledger must accept them
    root, _ = _status_root(tmp_path, "incomplete")
    assert check_ledger(root, username=None, denylist=None) == []


def test_ledger_flags_unknown_status(tmp_path):
    root, run = _status_root(tmp_path, "partial")
    found = check_ledger(root, username=None, denylist=None)
    assert _kinds(found) == [(run.relative_to(root).as_posix(), "bundle.json", "incomplete")]


def test_ledger_flags_committed_stored_file(tmp_path):
    root, run = _split_root(tmp_path)
    split(run, KEPT_RUN | {"bundle.json"})
    rel = next(r for r in json.loads((run / "bundle.json").read_text())["files"] if "quality" in r)
    (run / rel).parent.mkdir(exist_ok=True)
    (run / rel).write_text("{}\n")
    found = check_ledger(root, username=None, denylist=None)
    assert any(f.kind == "kept_mismatch" and f.file == rel for _, f in found)
    assert all(b == run.relative_to(root).as_posix() for b, _ in found)


def test_ledger_experiments_are_complete(tmp_path):
    root = tmp_path / "repo"
    run = make_bundle(
        root / "results" / "experiments" / "x" / "runs",
        label="A",
        weights="w",
        envelope="e",
        grades={"general/x": [True]},
    )
    (root / "results" / "layout.toml").write_text(SPLIT_LAYOUT)
    assert check_ledger(root, username=None, denylist=None) == []
    (run / "speed.jsonl").unlink()
    assert [f.kind for _, f in check_ledger(root, username=None, denylist=None)] == ["missing_file"]


def _withdraw(root, sha, bundles):
    (root / "results" / "withdrawn.toml").write_text(
        f'[[withdrawn]]\nsha256 = "{sha}"\ndate = "2026-11-03"\nreason = "leak"\n'
        f"bundles = {json.dumps(bundles)}\n"
    )


def test_ledger_flags_bad_withdrawn_row(tmp_path):
    root, run = _split_root(tmp_path)
    split(run, KEPT_RUN | {"bundle.json"})
    rel = run.relative_to(root).as_posix()
    _withdraw(root, "a" * 64, [rel])
    found = check_ledger(root, username=None, denylist=None)
    assert [(b, f.kind) for b, f in found] == [("results/withdrawn.toml", "withdrawn_row")]
    assert "does not list" in found[0][1].detail
    _withdraw(root, "a" * 64, ["results/2026-11-30/runs/nope"])
    found = check_ledger(root, username=None, denylist=None)
    assert [f.kind for _, f in found] == ["withdrawn_row"]
    assert "no such bundle" in found[0][1].detail


def test_ledger_accepts_good_withdrawn_row(tmp_path):
    root, run = _split_root(tmp_path)
    split(run, KEPT_RUN | {"bundle.json"})
    sha = json.loads((run / "bundle.json").read_text())["files"]["speed.jsonl"]
    _withdraw(root, sha, [run.relative_to(root).as_posix()])
    assert check_ledger(root, username=None, denylist=None) == []


def test_ledger_reports_bad_toml_as_finding(tmp_path):
    root, _ = _split_root(tmp_path)
    (root / "results" / "layout.toml").write_text("[[layout]\n")
    found = check_ledger(root, username=None, denylist=None)
    assert [(b, f.kind) for b, f in found] == [("results/layout.toml", "ledger")]
    (root / "results" / "layout.toml").write_text(SPLIT_LAYOUT)
    (root / "results" / "withdrawn.toml").write_text("[[withdrawn]]\nsha256 = 'x'\n")
    found = check_ledger(root, username=None, denylist=None)
    assert [(b, f.kind) for b, f in found] == [("results/withdrawn.toml", "ledger")]


def test_ledger_makes_no_requests(fake_store, tmp_path):
    root, run = _split_root(tmp_path)
    push_all(fake_store, run)
    split(run, KEPT_RUN | {"bundle.json"})
    fake_store.requests.clear()
    assert check_ledger(root, username=None, denylist=None) == []
    assert fake_store.requests == []


def _git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
    )


def test_changed_bundles_in_git_repo(tmp_path):
    root = tmp_path / "repo"
    a = make_bundle(
        root / "results" / "2026-11-02" / "runs",
        label="A",
        weights="w1",
        envelope="e",
        grades={"general/x": [True]},
    )
    b = make_bundle(
        root / "results" / "2026-11-02" / "runs",
        label="B",
        weights="w2",
        envelope="e",
        grades={"general/x": [False]},
    )
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "checkout", "-q", "-b", "feature")
    (b / "speed.jsonl").write_text("{}\n")
    (root / "README.md").write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "change")
    assert changed_bundles(root, "main") == [b]
    shutil.rmtree(a)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "delete")
    assert changed_bundles(root, "main") == [b]


def test_changed_without_base_fails_loudly(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "commit", "-q", "--allow-empty", "-m", "base")
    with pytest.raises(LedgerError, match="origin/main"):
        changed_bundles(root, "origin/main")
    with pytest.raises(LedgerError):
        changed_bundles(tmp_path / "not-a-repo", "main")


def test_cli_modes_exclusive(tmp_path, capsys):
    for argv in (
        ["verify", "--ledger", "--changed", "main"],
        ["verify", str(tmp_path), "--ledger"],
        ["verify", str(tmp_path), "--changed", "main"],
        ["verify"],
    ):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2


def test_cli_ledger_mode(tmp_path, capsys, monkeypatch):
    root, run = _split_root(tmp_path)
    split(run, KEPT_RUN | {"bundle.json"})
    assert cli.main(["verify", "--ledger", "--no-username", "--root", str(root)]) == 0
    (run / "stray.txt").write_text("x")
    assert cli.main(["verify", "--ledger", "--no-username", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert f"{run.relative_to(root).as_posix()}: stray.txt:0 unlisted_file" in out


def test_cli_changed_mode_fetches_from_store(fake_store, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "root")
    root, run = _split_root(tmp_path)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "checkout", "-q", "-b", "feature")
    push_all(fake_store, run)
    split(run, KEPT_RUN | {"bundle.json"})
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "split")
    argv = ["verify", "--changed", "main", "--root", str(root), "--no-username"]
    assert cli.main(argv) == 0
    assert any(m == "GET" for m, _, _ in fake_store.requests)
    files = json.loads((run / "bundle.json").read_text())["files"]
    del fake_store.objects[object_key(next(s for r, s in files.items() if "quality" in r))]
    shutil.rmtree(tmp_path / "cache")
    assert cli.main(argv) == 1
    assert cli.main(["verify", "--changed", "nope", "--root", str(root), "--no-username"]) == 1


def test_cli_changed_mode_accepts_incomplete_bundle(fake_store, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "root")
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "results" / "layout.toml").write_text(SPLIT_LAYOUT)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "checkout", "-q", "-b", "feature")
    for label, status in (("A", "incomplete"), ("B", "partial")):
        make_bundle(
            root / "results" / "2026-11-09" / "runs",
            label=label,
            weights="w",
            envelope="e",
            grades={"general/x": [True]},
            status=status,
        )
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", label)
        argv = ["verify", "--changed", "main", "--root", str(root), "--no-username"]
        assert cli.main(argv) == (0 if status == "incomplete" else 1)


# ---- store audit ----


def _audit_root(tmp_path, fake_store):
    root, run = _split_root(tmp_path)
    push_all(fake_store, run)
    split(run, KEPT_RUN | {"bundle.json"})
    return root, run


def test_audit_no_split_weeks(tmp_path, capsys):
    (tmp_path / "results").mkdir()
    assert cli.main(["store", "audit", "--root", str(tmp_path)]) == 0
    assert "no split weeks" in capsys.readouterr().out
    (tmp_path / "results" / "layout.toml").write_text(
        '[[layout]]\nfrom = "2026-11-02"\nkeep = "all"\n'
    )
    assert cli.main(["store", "audit", "--root", str(tmp_path), "--rehash", "50"]) == 0
    assert "no split weeks" in capsys.readouterr().out


def test_audit_ok_and_seed_is_printed(fake_store, tmp_path, capsys):
    root, run = _audit_root(tmp_path, fake_store)
    argv = ["store", "audit", "--root", str(root), "--rehash", "2", "--seed", "2026-W45"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "seed 2026-W45" in out


def test_audit_default_seed_is_iso_week(fake_store, tmp_path, capsys, monkeypatch):
    root, _ = _audit_root(tmp_path, fake_store)
    assert cli.main(["store", "audit", "--root", str(root)]) == 0
    assert "seed 20" in capsys.readouterr().out
    assert cli._iso_week_seed(datetime(2026, 11, 4, tzinfo=UTC)) == "2026-W45"


def test_audit_missing_object_fails(fake_store, tmp_path, capsys):
    root, run = _audit_root(tmp_path, fake_store)
    files = json.loads((run / "bundle.json").read_text())["files"]
    rel = next(r for r in files if "quality" in r)
    del fake_store.objects[object_key(files[rel])]
    assert cli.main(["store", "audit", "--root", str(root)]) == 1
    assert "rehashed 0" in capsys.readouterr().out
    assert cli.main(["store", "audit", "--root", str(root), "--rehash", "99"]) == 1
    assert "rehashed 2" in capsys.readouterr().out
    assert cli.main(["store", "audit", "--root", str(root)]) == 1
    assert f"missing {files[rel]} " in (out := capsys.readouterr().out)
    assert out.split("\n")[0].endswith(f"/{rel}")


def test_audit_reports_withdrawn_not_missing(fake_store, tmp_path, capsys):
    root, run = _audit_root(tmp_path, fake_store)
    files = json.loads((run / "bundle.json").read_text())["files"]
    rel = next(r for r in files if "quality" in r)
    del fake_store.objects[object_key(files[rel])]
    _withdraw(root, files[rel], [run.relative_to(root).as_posix()])
    assert cli.main(["store", "audit", "--root", str(root), "--rehash", "99"]) == 0
    out = capsys.readouterr().out
    assert f"withdrawn {files[rel]}" in out and "missing" not in out


def test_audit_rehash_is_deterministic_and_total(fake_store, tmp_path, capsys):
    root, run = _audit_root(tmp_path, fake_store)
    second = make_bundle(
        root / "results" / "2026-11-30" / "judge",
        label="B",
        weights="w2",
        envelope="e",
        grades={"general/x": [True]},
    )
    push_all(fake_store, second)
    split(second, {"bundle.json"})

    def gets(seed):
        fake_store.requests.clear()
        argv = ["store", "audit", "--root", str(root), "--rehash", "3", "--seed", seed]
        assert cli.main(argv) == 0
        return sorted(p for m, p, _ in fake_store.requests if m == "GET")

    first = gets("2026-W45")
    assert len(first) == 3
    assert gets("2026-W45") == first
    assert any(gets(f"2026-W{n}") != first for n in range(46, 52))


def test_audit_catches_corrupt_object(fake_store, tmp_path, capsys):
    root, run = _audit_root(tmp_path, fake_store)
    files = json.loads((run / "bundle.json").read_text())["files"]
    fake_store.objects[object_key(files["speed.jsonl"])] = b"corrupt"
    assert cli.main(["store", "audit", "--root", str(root), "--rehash", "99"]) == 1
    assert f"corrupt {files['speed.jsonl']}" in capsys.readouterr().out
