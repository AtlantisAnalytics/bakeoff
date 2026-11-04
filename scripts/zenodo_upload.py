"""Upload a built yearly archive to Zenodo and print the DOI.

Usage: zenodo_upload.py OUT_DIR --year YEAR

The token comes only from ZENODO_TOKEN; the base URL from ZENODO_URL (default zenodo.org).
"""

import argparse
import os
import sys
from pathlib import Path

from bakeoff.archive import (
    DEFAULT_ZENODO,
    ArchiveError,
    load_archive_rows,
    zenodo_publish,
)

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="zenodo_upload")
    p.add_argument("out_dir")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--root", default=str(REPO))
    args = p.parse_args(argv)
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("error: ZENODO_TOKEN is not set", file=sys.stderr)
        return 1
    try:
        rows = load_archive_rows(Path(args.root))
        if args.year in rows:
            raise ArchiveError(f"archive.toml already has a record for {args.year}")
        prev = rows.get(args.year - 1)
        base = os.environ.get("ZENODO_URL") or DEFAULT_ZENODO
        print(
            zenodo_publish(Path(args.out_dir), args.year, prev.doi if prev else None, base, token)
        )
    except ArchiveError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
