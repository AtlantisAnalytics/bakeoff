"""Build the public GSM8K and IFEval slices from pinned upstream files.

Usage: uv run --project <repo> python scripts/make_slices.py [--cache DIR] [--out DIR]

Downloaded files are data: they are parsed as JSON lines only, cached under runs/ (git-ignored),
and refused when their SHA-256 differs from SOURCES. Rows whose text fails bundle.scan_text are
dropped before sampling, and the drop count is recorded in the manifest.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import httpx

from bakeoff.bundle import scan_text
from bakeoff.ifeval import SUPPORTED

REPO = Path(__file__).resolve().parents[1]
SEED = 20261013

SOURCES = {
    "gsm8k": {
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/{rev}/grade_school_math/data/test.jsonl",
        "license_url": "https://github.com/openai/grade-school-math/blob/{rev}/LICENSE",
        "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "file": "gsm8k_test.jsonl",
    },
    "ifeval": {
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "url": "https://huggingface.co/datasets/google/IFEval/resolve/{rev}/ifeval_input_data.jsonl",
        "license_url": "https://huggingface.co/datasets/google/IFEval/blob/{rev}/README.md",
        "sha256": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        "file": "ifeval_input_data.jsonl",
    },
}

GSM8K_SYSTEM = "Solve the problem. End your reply with a final line of the form 'Answer: <number>'."

ATTRIBUTION = {
    "gsm8k": """# Attribution: GSM8K slice

- Dataset: GSM8K (grade-school math word problems), test split.
- Authors: Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
  Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse and John
  Schulman (OpenAI).
- Paper: Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems",
  https://arxiv.org/abs/2110.14168
- License: MIT. License text: {license_url}
- Source file (pinned): {url}
- Source SHA-256: {sha256}

This slice is unmodified apart from sampling ({n} of {pool} eligible rows, seeded, after
dropping {dropped} rows that failed the scrub scan) and the field mapping: `question` becomes
`prompt`, and `expected` is the number after `####` with thousands commas removed.
""",
    "ifeval": """# Attribution: IFEval slice

- Dataset: IFEval (Instruction-Following Eval), `ifeval_input_data.jsonl`.
- Authors: Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan,
  Denny Zhou and Le Hou (Google).
- Paper: Zhou et al. 2023, "Instruction-Following Evaluation for Large Language Models",
  https://arxiv.org/abs/2311.07911
- License: Apache-2.0 (dataset card `license: apache-2.0`). Card: {license_url}
- Source file (pinned): {url}
- Source SHA-256: {sha256}

This slice is unmodified apart from sampling ({n} of {pool} eligible rows, seeded, where eligible
means every instruction is implemented by our checker subset, after dropping {dropped} rows that
failed the scrub scan) and the field mapping: `prompt` is kept, `expected` holds
`instruction_id_list` and `kwargs` with null-valued keys dropped.
""",
}


def fetch(name: str, cache: Path) -> Path:
    src = SOURCES[name]
    path = cache / src["file"]
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        r = httpx.get(src["url"].format(rev=src["revision"]), follow_redirects=True, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != src["sha256"]:
        raise SystemExit(f"{name}: SHA-256 mismatch: expected {src['sha256']}, got {got}")
    return path


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def gsm8k_items(rows: list[dict]) -> tuple[list[dict], int]:
    pool, dropped = [], 0
    for idx, row in enumerate(rows):
        answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        if not re.fullmatch(r"-?\d+(\.\d+)?", answer):
            raise SystemExit(f"gsm8k row {idx}: answer {answer!r} is not a plain number")
        if scan_text(row["question"] + "\n" + row["answer"], []):
            dropped += 1
            continue
        pool.append({"id": f"gsm8k-test-{idx}", "prompt": row["question"], "expected": answer})
    return pool, dropped


def ifeval_items(rows: list[dict]) -> tuple[list[dict], int]:
    pool, dropped = [], 0
    for row in rows:
        ids = row["instruction_id_list"]
        if not ids or not set(ids) <= SUPPORTED:
            continue
        kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in row["kwargs"]]
        text = row["prompt"] + "\n" + json.dumps(kwargs, ensure_ascii=False)
        if scan_text(text, []):
            dropped += 1
            continue
        expected = {"instruction_id_list": ids, "kwargs": kwargs}
        pool.append({"id": f"ifeval-{row['key']}", "prompt": row["prompt"], "expected": expected})
    return pool, dropped


def q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def manifest_text(name: str, license_: str, n_pool: int, dropped: int, system: str | None) -> str:
    src = SOURCES[name]
    rev = src["revision"]
    lines = [
        'track = "general"',
        f"name = {q(name)}",
        "version = 1",
        'split = "public"',
        f"license = {q(license_)}",
        f"grader = {q('numeric' if name == 'gsm8k' else 'ifeval')}",
        "max_tokens = 1024",
    ]
    if system:
        lines.append(f"system = {q(system)}")
    lines += [
        "",
        "[source]",
        f"url = {q(src['url'].format(rev=rev))}",
        f"revision = {q(rev)}",
        f"sha256 = {q(src['sha256'])}",
        f"license_url = {q(src['license_url'].format(rev=rev))}",
        f"pool_size = {n_pool}",
        f"dropped_scrub = {dropped}",
        "",
    ]
    return "\n".join(lines)


def write_suite(out: Path, name: str, license_: str, pool: list[dict], dropped: int, k: int):
    picked = random.Random(SEED).sample(range(len(pool)), min(k, len(pool)))
    items = [pool[i] for i in sorted(picked)]
    d = out / "general" / name
    d.mkdir(parents=True, exist_ok=True)
    system = GSM8K_SYSTEM if name == "gsm8k" else None
    (d / "manifest.toml").write_text(
        manifest_text(name, license_, len(pool), dropped, system), encoding="utf-8"
    )
    (d / "items.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8"
    )
    src = SOURCES[name]
    rev = src["revision"]
    (d / "ATTRIBUTION.md").write_text(
        ATTRIBUTION[name].format(
            url=src["url"].format(rev=rev),
            license_url=src["license_url"].format(rev=rev),
            sha256=src["sha256"],
            n=len(items),
            pool=len(pool),
            dropped=dropped,
        ),
        encoding="utf-8",
    )
    print(f"{name}: {len(items)} items (pool {len(pool)}, dropped_scrub {dropped})")
    return len(items)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=REPO / "runs" / ".cache" / "slices")
    ap.add_argument("--out", type=Path, default=REPO / "suites")
    args = ap.parse_args(argv)

    pool, dropped = gsm8k_items(read_rows(fetch("gsm8k", args.cache)))
    write_suite(args.out, "gsm8k", "MIT", pool, dropped, 100)
    pool, dropped = ifeval_items(read_rows(fetch("ifeval", args.cache)))
    n = write_suite(args.out, "ifeval", "Apache-2.0", pool, dropped, 100)
    if n < 50:
        print(f"error: only {n} IFEval rows survived the filter", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
