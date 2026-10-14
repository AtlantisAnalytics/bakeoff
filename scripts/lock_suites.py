"""Write suites/SUITES.lock.json: {key: {version: hash}}, keeping old versions' entries."""

import json
import sys

from bakeoff.suites import SuiteError, resolve_suites, suites_root


def main() -> int:
    root = suites_root()
    lock_path = root / "SUITES.lock.json"
    lock: dict = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
    try:
        found = resolve_suites([str(root)]) if any(root.rglob("manifest.toml")) else []
    except SuiteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for s in found:
        lock.setdefault(s.key, {})[str(s.version)] = s.hash
    lock_path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n")
    print(f"locked {len(found)} suite(s) in {lock_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
