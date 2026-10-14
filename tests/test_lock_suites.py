import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lock_suites", ROOT / "scripts" / "lock_suites.py")
lock_suites = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock_suites)

DEMO = ROOT / "tests" / "data" / "suites" / "demo"


def _setup(tmp_path):
    root = tmp_path / "suites"
    shutil.copytree(DEMO, root / "demo")
    return root, tmp_path / "lock.json"


def test_empty_root_writes_empty_lock(tmp_path):
    root = tmp_path / "suites"
    root.mkdir()
    lock = tmp_path / "lock.json"
    assert lock_suites.main(root, lock) == 0
    assert json.loads(lock.read_text()) == {}


def test_writes_and_keeps_old_versions(tmp_path):
    root, lock = _setup(tmp_path)
    lock.write_text(json.dumps({"demo/exact": {"0": "old"}}))
    assert lock_suites.main(root, lock) == 0
    data = json.loads(lock.read_text())
    assert data["demo/exact"]["0"] == "old"
    assert len(data["demo/exact"]["1"]) == 64


def test_refuses_unbumped_edit(tmp_path, capsys):
    root, lock = _setup(tmp_path)
    assert lock_suites.main(root, lock) == 0
    before = lock.read_text()
    (root / "demo" / "items.jsonl").write_text('{"id": "z", "prompt": "q", "expected": "q"}\n')
    assert lock_suites.main(root, lock) == 1
    assert "bump" in capsys.readouterr().err
    assert lock.read_text() == before
