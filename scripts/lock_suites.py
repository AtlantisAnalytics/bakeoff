"""Write suites/SUITES.lock.json: {key: {version: hash}}, keeping old versions' entries.

Refuses (exit 1, nothing written) when a locked version's hash differs: bump `version` instead.
"""

import json
import sys
from pathlib import Path

from bakeoff.suites import SuiteError, check_lock, resolve_suites, suites_root


def main(root: Path | None = None, lock_path: Path | None = None) -> int:
    root = Path(root) if root else suites_root()
    lock_path = Path(lock_path) if lock_path else root / "SUITES.lock.json"
    lock: dict = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else {}
    try:
        found = resolve_suites([str(root)]) if any(root.rglob("manifest.toml")) else []
    except SuiteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    conflicts = [s for s in found if lock.get(s.key, {}).get(str(s.version), s.hash) != s.hash]
    if conflicts:
        for problem in check_lock(conflicts, lock):
            print(f"error: {problem}", file=sys.stderr)
        return 1
    for s in found:
        lock.setdefault(s.key, {})[str(s.version)] = s.hash
    lock_path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"locked {len(found)} suite(s) in {lock_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
