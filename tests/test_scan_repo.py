"""scripts/scan_repo.py: the tip-of-tree scrub scan CI runs."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scan_repo.py"


@pytest.fixture
def scan():
    spec = importlib.util.spec_from_file_location("scan_repo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    files = {
        "a.txt": "fine\nip 10.1.2.3\n",  # scrub-ok
        "b.txt": "allowed 10.9.9.9  # scrub-ok\n",  # scrub-ok
        "vendor/c.txt": "ip 10.4.4.4\n",  # scrub-ok
        "d.bin": "ip 10.5.5.5\0\n",  # scrub-ok
        "e.txt": "the secret-term is here\n",
    }
    for rel, text in files.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text, encoding="utf-8")
    (r / "untracked.txt").write_text("ip 10.6.6.6\n", encoding="utf-8")  # scrub-ok
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "add", "-f", *files], check=True)  # past global ignores
    return r


def test_findings_skips_and_exclusions(scan, repo):
    findings, skipped = scan.scan_repo(repo, ["secret-term"], None, ["vendor/"])
    assert findings == [
        ("a.txt", 2, "private_ip", "10.1.2.3"),  # scrub-ok
        ("e.txt", 1, "denylist", "secret-term"),
    ]
    assert skipped == [("b.txt", 1, "private_ip")]  # binary, excluded and untracked never read


def test_main_lists_skipped_lines_without_their_match(scan, repo, monkeypatch, capsys):
    monkeypatch.setattr(scan, "REPO", repo)
    assert scan.main(["--exclude", "vendor", "--exclude", "a.txt", "--exclude", "e.txt"]) == 0
    out = capsys.readouterr().out
    assert out == "skipped (scrub-ok): b.txt:1 private_ip\n"
    assert "10.9.9.9" not in out  # scrub-ok
    assert scan.main(["secret-term"]) == 1
    assert "e.txt:1 denylist secret-term" in capsys.readouterr().out
