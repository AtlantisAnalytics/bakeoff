"""Scrub scan over every tracked text file (the tip-of-tree check; M5 extends it to history).

Usage: scan_repo.py [--exclude PATH]... [--username NAME] [TERM...]

Runs bundle.scan_text over each file from `git ls-files`, skips lines containing `scrub-ok`,
prints `path:line kind match` and exits 1 on any finding. Each skipped line is listed as
`skipped (scrub-ok): path:line kind`, without the match, so a marker can be audited and a private
term never lands in a log. Private terms are passed as arguments and are never stored anywhere.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from bakeoff.bundle import scan_text

MARKER = "scrub-ok"
REPO = Path(__file__).resolve().parents[1]


def tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def scan_repo(
    repo: Path, terms: list[str], username: str | None, exclude: list[str]
) -> tuple[list[tuple[str, int, str, str]], list[tuple[str, int, str]]]:
    """The findings, and the hits on `scrub-ok` lines (path, line, kind)."""
    skip = [e.rstrip("/") for e in exclude]
    findings, skipped = [], []
    for rel in tracked_files(repo):
        if any(rel == e or rel.startswith(e + "/") for e in skip):
            continue
        path = repo / rel
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue  # binary
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.split("\n")
        for no, kind, match in scan_text(text, terms, username):
            if MARKER in lines[no - 1]:
                skipped.append((rel, no, kind))
            else:
                findings.append((rel, no, kind, match))
    return findings, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scan_repo")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--username")
    p.add_argument("terms", nargs="*")
    args = p.parse_args(argv)
    findings, skipped = scan_repo(REPO, args.terms, args.username, args.exclude)
    for rel, no, kind in skipped:
        print(f"skipped (scrub-ok): {rel}:{no} {kind}")
    for rel, no, kind, match in findings:
        print(f"{rel}:{no} {kind} {match}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
