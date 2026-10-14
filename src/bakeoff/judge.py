"""Blind, order-debiased pairwise judge over two bundles' open-ended outputs.

Budget guarantee: before every call, `spent + that call's worst case <= max_usd` is checked, with
input tokens estimated pessimistically as UTF-8 bytes / 2 and output capped at JUDGE_MAX_TOKENS.
Actual reported usage is charged after each call (worst case if none is reported, or if the call
raises). So `cost_usd <= max_usd + the excess of one call's actual usage over its worst case`;
that excess is zero unless the provider bills more tokens than the estimate allows. The CLI builds
the client without retries, so one call is one billed attempt.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bakeoff.bundle import verify_bundle, write_bundle
from bakeoff.client import EndpointClient, split_think
from bakeoff.stats import wilson
from bakeoff.suites import SuiteError, load_suite, suites_root

JUDGE_SYSTEM = (
    "You are an impartial judge comparing two answers to the same question. Judge helpfulness, "
    "correctness and clarity for the question. Ignore length and style flourishes. "
    'Reply only with JSON: {"winner": "1" | "2" | "tie", "reason": "<one sentence>"}'
)
JUDGE_MAX_TOKENS = 400
FATAL_KINDS = frozenset(
    {"missing_file", "hash_mismatch", "unlisted_file", "bundle_hash", "unreadable"}
)
REASON_MAX = 300
WINNERS = ("1", "2", "tie")


class JudgeError(Exception):
    pass


@dataclass(frozen=True)
class Prices:
    in_per_mtok: float
    out_per_mtok: float


def per_call_worst_case(text: str, prices: Prices) -> float:
    """Worst-case USD for one call whose full prompt text (system included) is `text`."""
    tokens_in = len(text.encode("utf-8")) / 2
    return (tokens_in * prices.in_per_mtok + JUDGE_MAX_TOKENS * prices.out_per_mtok) / 1e6


def estimate_usd(prompts: list[str], prices: Prices, calls_per_item: int = 2) -> float:
    return sum(per_call_worst_case(p, prices) for p in prompts) * calls_per_item


def combine(first: str | None, second: str | None) -> str:
    """First call has A as "1"; second call has B as "1". A win needs both orders to agree."""
    if first == "1" and second == "2":
        return "a"
    if first == "2" and second == "1":
        return "b"
    return "tie"


def parse_reply(text: str) -> tuple[str | None, str]:
    """(winner or None, reason). The first {...} that parses with a valid winner wins."""
    dec = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = dec.raw_decode(text, pos)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            w = obj.get("winner")
            w = (
                str(w).strip().lower()
                if isinstance(w, str | int) and not isinstance(w, bool)
                else ""
            )
            if w in WINNERS:
                reason = obj.get("reason")
                return w, (reason if isinstance(reason, str) else "")[:REASON_MAX]
        pos = text.find("{", pos + 1)
    return None, text[:REASON_MAX]


def _user_message(question: str, one: str, two: str) -> str:
    return f"Question:\n{question}\n\nAnswer 1:\n{one}\n\nAnswer 2:\n{two}"


def _question(messages: list[dict]) -> str:
    return "\n\n".join(str(m["content"]) for m in messages if m.get("role") != "system")


def _load_side(path: Path, suite_key: str, expected_hash: str) -> tuple[str, dict[str, dict]]:
    """Verify one input bundle's hashes and return (bundle_sha256, records by id)."""
    path = Path(path)
    bad = [f for f in verify_bundle(path, None, None) if f.kind in FATAL_KINDS]
    if bad:
        shown = "; ".join(f"{f.file}: {f.kind}" for f in bad[:3])
        raise JudgeError(f"bundle failed verification ({shown})")
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    entry = next((s for s in meta.get("suites", []) if s.get("key") == suite_key), None)
    if entry is None:
        raise JudgeError(f"a bundle has no {suite_key} suite")
    if entry.get("hash") != expected_hash:
        raise JudgeError(f"a bundle's {suite_key} suite hash differs from the current suite")
    rel = f"quality/{suite_key.replace('/', '.')}.jsonl"
    f = path / rel
    if not f.is_file():
        raise JudgeError(f"a bundle has no {rel}")
    recs: dict[str, dict] = {}
    for line in f.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                r = json.loads(line)
                recs[r["id"]] = r
            except (ValueError, KeyError, TypeError) as e:
                raise JudgeError(f"a bundle's {rel} is malformed") from e
    return str(meta["bundle_sha256"]), recs


async def run_judge(
    bundle_a: Path,
    bundle_b: Path,
    client: EndpointClient,
    judge_model: str,
    prices: Prices,
    max_usd: float,
    allow_partial: bool,
    out_root: Path,
    suite_key: str = "general/open-ended",
) -> Path:
    try:
        suite = load_suite(suites_root() / suite_key)
    except SuiteError as e:
        raise JudgeError(f"cannot load suite {suite_key}: {e}") from e
    hash_a, recs_a = _load_side(bundle_a, suite_key, suite.hash)
    hash_b, recs_b = _load_side(bundle_b, suite_key, suite.hash)
    out_dir = Path(out_root) / f"judge-{hash_a[:8]}-vs-{hash_b[:8]}"
    if out_dir.exists():
        raise JudgeError("output directory already exists; not overwriting")

    todo = []
    for item in sorted(suite.items, key=lambda i: i.id):
        ra, rb = recs_a.get(item.id), recs_b.get(item.id)
        if not ra or not rb or ra.get("error") is not None or rb.get("error") is not None:
            continue
        if not isinstance(ra.get("output"), str) or not isinstance(rb.get("output"), str):
            continue
        todo.append(
            (
                item.id,
                _question(item.messages),
                split_think(ra["output"])[0],
                split_think(rb["output"])[0],
            )
        )
    skipped = len(suite.items) - len(todo)

    # per item, the two user messages (second order swaps the answers; same length as the first)
    plans = [[_user_message(q, xa, xb), _user_message(q, xb, xa)] for _, q, xa, xb in todo]
    item_worsts = [
        [per_call_worst_case(JUDGE_SYSTEM + "\n" + u, prices) for u in users] for users in plans
    ]
    full = [JUDGE_SYSTEM + "\n" + users[0] for users in plans]
    estimate = estimate_usd(full, prices)
    if estimate > max_usd and not allow_partial:
        raise JudgeError(
            f"worst-case estimate ${estimate:.4f} is over --max-usd ${max_usd:.4f}; "
            "raise the cap or use --allow-partial"
        )

    counts = {"a": 0, "b": 0, "tie": 0}
    items: list[dict] = []
    spent = 0.0
    status, reason = "complete", None
    interrupted: BaseException | None = None

    errored = False

    async def call(user: str, worst: float) -> tuple[str | None, str, float] | None:
        nonlocal spent, errored
        if spent + worst > max_usd:
            return None
        try:
            r = await client.chat(
                [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                JUDGE_MAX_TOKENS,
                temperature=0.0,
                seed=0,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception:  # noqa: BLE001 - charge the worst case and stop; never leak the text
            spent += worst
            errored = True
            return None, "call failed", worst
        if r.prompt_tokens is not None and r.completion_tokens is not None:
            cost = r.prompt_tokens * prices.in_per_mtok / 1e6
            cost += r.completion_tokens * prices.out_per_mtok / 1e6
        else:
            cost = worst
        spent += cost
        if r.error is not None:
            return None, "call failed", cost
        winner, why = parse_reply(r.content)
        return winner, why, cost

    try:
        for (iid, *_), users, worsts in zip(todo, plans, item_worsts, strict=True):
            if spent + sum(worsts) > max_usd:
                status, reason = "incomplete", "budget"
                break
            outs = []
            for u, w in zip(users, worsts, strict=True):
                got = await call(u, w)
                if got is None or errored:
                    if got is not None:
                        outs.append(got)
                    break
                outs.append(got)
            if len(outs) == 2 and not errored:
                (w1, why1, c1), (w2, why2, c2) = outs
                res = combine(w1, w2)
                counts[res] += 1
                items.append(
                    {
                        "id": iid,
                        "first": w1,
                        "second": w2,
                        "result": res,
                        "reasons": [why1, why2],
                        "cost_usd": c1 + c2,
                    }
                )
                continue
            if outs:  # stopped mid-item: keep its cost, count it nowhere
                items.append(
                    {
                        "id": iid,
                        "first": outs[0][0],
                        "second": outs[1][0] if len(outs) > 1 else None,
                        "result": None,
                        "reasons": [o[1] for o in outs],
                        "cost_usd": sum(o[2] for o in outs),
                    }
                )
            status, reason = ("incomplete", "error" if errored else "budget")
            break
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        status, reason = "incomplete", "interrupted"
        interrupted = exc

    decided = counts["a"] + counts["b"]
    meta = {
        "bundle_format": "judge-m1",
        "status": status,
        "status_reason": reason,
        "a": hash_a,
        "b": hash_b,
        "suite": {"key": suite_key, "hash": suite.hash},
        "judge_model": judge_model,
        "prices": asdict(prices),
        "max_usd": max_usd,
        "estimate_usd": estimate,
        "cost_usd": spent,
        "counts": counts,
        "items_skipped": skipped,
        "win_rate_a": counts["a"] / decided if decided else None,
        "ci95": list(wilson(counts["a"], decided)) if decided else None,
    }
    write_bundle(out_dir, meta, {"items.jsonl": items})
    if interrupted is not None:
        raise interrupted
    return out_dir
