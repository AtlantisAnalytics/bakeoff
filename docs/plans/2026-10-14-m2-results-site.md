# M2 Results and Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `bakeoff publish week.toml` turns run bundles into a published week, with rule-based
verdicts, `week.json`, `picks.json` and a generated static site, JSON API and feed. `bakeoff site`
regenerates and checks `site/`. CI gates PRs.

**Architecture:** Pure layers, each with its own module.
1. Input loaders (`weekcfg.py`).
2. A read-only view of a bundle (`runview.py`).
3. Deterministic resampling (`resample.py`).
4. Verdict rules (`verdicts.py`).
5. The week document and its schema (`week.py`).
6. The orchestrator (`publish.py`), which writes `results/`.
7. A site generator (`web/`), which reads only the committed `week.json` files and returns
   `{path: bytes}`. That keeps `--check` a simple byte comparison.

**Tech Stack:** Python ≥ 3.11 standard library only (`tomllib`, `html`, `xml.sax.saxutils`,
`hashlib`, `shutil`), on top of the M1 package. Development uses uv, pytest and ruff. CI is GitHub
Actions.

**Spec:** [docs/specs/2026-10-14-m2-results-site-design.md](../specs/2026-10-14-m2-results-site-design.md).
Read it before starting any task. The M1 spec and code are the base.

## Global Constraints

- **Tooling (unchanged from M1):**
  - Ruff: `line-length = 100`, `target-version = "py311"`, `select = ["E", "F", "I", "UP", "B"]`.
  - Async code is tested with `asyncio.run(...)`; there is no pytest-asyncio.
- **Dependencies:** none new at runtime. `httpx` stays the only runtime dependency.
- **Commands:** `<repo>` is the absolute path to the checkout. Use absolute paths, `git -C` and
  `uv run --project`; never `cd X && …`.
  - `uv run --project <repo> pytest -q`
  - `uv run --project <repo> ruff check`
  - `uv run --project <repo> ruff format --check`
- **The generator package is `src/bakeoff/web/`, not `src/bakeoff/site/`** (plan ruling).
  `pyproject.toml` has `extend-exclude = ["site"]` for ruff, which would silently stop the package
  from being linted.
- **Scrub rules (public repo).** Committed files must not contain:
  - internal hostnames, private or LAN IPs, or real `host:port` pairs;
  - usernames or local paths.

  Fixtures use fictional values: hostname `lab-box-7`, username `alice`, and documentation IPs only.
  A line that plants a violation on purpose carries `scrub-ok`.
- **Deterministic output:**
  - Nothing reads the clock, the environment or the locale while generating output.
  - Every timestamp comes from `week.toml`.
  - JSON outputs are written with `json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"` and encoded as UTF-8.
- **HTML pages:**
  - Every interpolated value passes through `html.escape(str(v), quote=True)`.
  - Atom XML uses `xml.sax.saxutils.escape` (with `quoteattr` for attributes).
  - There is no `<script>`, no inline `style=` attribute and no `<style>` element anywhere in the
    generated pages.
- **URLs in pages** (`source_url`, `license_url`, `repo_url`, `base_url`, the newsletter `action`)
  must start with `https://`. Loaders reject any other URL.
- **Placeholder data.** `runs/` is git-ignored, and no real week is committed in M2. `results/`
  gains only `picks.json`, as `{}`, and only if the generator needs it; it doesn't.
- **Git:**
  - Work stays on the `m2-results-site` branch, and nothing is pushed without asking.
  - Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Hostile or odd model metadata.**
   - Example: a label like `Qwen3 "Instruct" <beta> & co`, with non-ASCII characters.
   - It must render escaped in HTML text, in HTML attributes and in Atom, and round-trip unchanged
     in JSON.
   - Pinned in Tasks 9 and 10.
2. **Partial bundles.** Bundles from `bakeoff run --only quality` (no speed or fit), or with
   `status: incomplete`, must load. They are listed as ineligible with reasons, never crash. Pinned
   in Task 5.
3. **Suite changes between weeks.** The pick's re-run uses a newer suite version (a different hash)
   than the candidate. The result is `not_comparable`, with verdicts capped at `watch`, and nothing
   crashes. Pinned in Task 6.
4. **Unsafe URL schemes.** `javascript:` or `http:` in `source_url` or `license_url` is rejected at
   load time, naming the field. Pinned in Task 4.
5. **Interrupted publish.** If `publish` dies partway through (a scrub finding, Ctrl-C), no
   half-written `results/<week>/` may remain, and a rerun must succeed. Pinned in Task 8.

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `src/bakeoff/runner.py`, `src/bakeoff/cli.py`, `src/bakeoff/bundle.py`, `docs/specs/2026-10-13-m1-harness-design.md` | M1 follow-ups | 1 |
| `src/bakeoff/graders.py`, `src/bakeoff/quality.py`, `src/bakeoff/runner.py` | Lenient JSON score; `bundle_format = "m2"` | 2 |
| `src/bakeoff/resample.py` | splitmix64, the verdict seed, the paired bootstrap CI | 3 |
| `src/bakeoff/weekcfg.py`, `tiers.toml`, `site.toml` | Loaders for tiers, `week.toml` and `site.toml` | 4 |
| `src/bakeoff/runview.py` | Read-only view of one run bundle; eligibility | 5 |
| `src/bakeoff/verdicts.py` | Rules v1: verdicts, picks and overrides for one tier | 6 |
| `src/bakeoff/week.py`, `src/bakeoff/schemas/week.v1.json`, `src/bakeoff/graders.py` (schema subset) | The week document and its JSON Schema | 7 |
| `src/bakeoff/publish.py` | Orchestration, bundle copying, corrections and `results/picks.json` | 8 |
| `src/bakeoff/web/fmt.py`, `src/bakeoff/web/pages.py`, `site/style.css`, `site_src/method.html`, `site_src/privacy.html` | Pages and display formatting | 9 |
| `src/bakeoff/web/build.py`, `src/bakeoff/cli.py` | API files, feed, `_headers`, launch gate, check, preview; the `publish` and `site` commands | 10 |
| `.github/workflows/ci.yml`, `README.md`, `PLAN.md`, `.gitignore` | CI, docs, the cross-check and the M2 check | 11 |

---

### Task 1: M1 follow-ups

**Files:**
- Modify: `src/bakeoff/runner.py` (`make_redactor`), `src/bakeoff/cli.py` (`_public_suite_text`,
  `pick_username`, `_cmd_verify`), `src/bakeoff/bundle.py` (`verify_bundle`),
  `docs/specs/2026-10-13-m1-harness-design.md` (the `run.toml` block)
- Test: `tests/test_runner.py`, `tests/test_cli.py`, `tests/test_bundle.py`

**Interfaces:**
- Produces:
  - `verify_bundle(run_dir, username, denylist, username_skip_output: bool = False) -> list[Finding]`
  - `pick_username(no_username, public_text=...) -> tuple[str | None, str | None, bool]`, where the
    third element is `skip_output`.
  - Task 8 calls both.

- [ ] **Step 1: Write the failing tests.**
  - `test_redactor_ignores_short_and_placeholder_keys`: for each key in `ollama`, `EMPTY`, `none`,
    `sk-no-key-required` and `abc1234` (7 characters), `make_redactor("http://192.0.2.5:11434/v1", (key,))("runtime ollama EMPTY none abc1234")` leaves those words intact. `# scrub-ok`
  - `test_redactor_still_redacts_real_key`: an 8+ character key is replaced by `<endpoint>`.
  - `test_public_suite_text_survives_any_error`: monkeypatch `resolve_suites` to raise
    `RuntimeError`. Expect `_public_suite_text()` to return `""`.
  - `test_username_skip_scopes_to_output`:
    1. Write a bundle whose `quality/x.y.jsonl` record has `"output": "alice went home"` and whose
       `bundle.json` `config.model.label` contains `alice`.
    2. `verify_bundle(dir, "alice", None, username_skip_output=True)` reports exactly one
       `username` finding, in `bundle.json`.
    3. With `False`, it reports two.
  - `test_pick_username_returns_skip_output`: when the name occurs in the public text, the result is
    `(name, notice, True)`. The scan is no longer skipped entirely.
- [ ] **Step 2: Run them.** `uv run --project <repo> pytest -q tests/test_runner.py tests/test_cli.py tests/test_bundle.py`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
  - `make_redactor` drops a secret when `len(s) < 8` or `s.lower()` is in
    `{"ollama", "empty", "none", "sk-no-key-required"}`.
  - `_public_suite_text` catches `Exception`.
  - In `verify_bundle`, `username_skip_output=True` leaves the username pattern out only for string
    values whose key is `output` in JSONL records. The other patterns still apply to those values.
  - `_cmd_verify` passes the flag through.
  - In the M1 spec's `run.toml` block, `suites` uses bare names, matching `examples/run.toml`.
- [ ] **Step 4: Run the full suite and ruff.** Expected: everything passes, including the existing
  `pick_username` tests, updated for the 3-tuple.
- [ ] **Step 5: Commit**: `"M1 follow-ups: placeholder keys, suite-text catch, username scope"`

### Task 2: Lenient JSON score

**Files:**
- Modify: `src/bakeoff/graders.py`, `src/bakeoff/quality.py`, `src/bakeoff/runner.py`
  (`bundle_format`)
- Test: `tests/test_graders.py`, `tests/test_quality.py`

**Interfaces:**
- Produces:
  - `Grade.extractable: bool | None = None`, a new last field with a default.
  - Quality records gain `"extractable": bool | None`.
  - Summaries of `json_schema` suites gain `extractable_accuracy: float | None` and
    `extractable_ci95: list[float] | None`.
  - `bundle_format = "m2"`. Task 5 reads all of these.

- [ ] **Step 1: Write the failing tests** in `tests/test_graders.py`.
  - With a schema `{"type": "object", "required": ["a"]}`:

    | Output | strict `passed` | `extractable` |
    |---|---|---|
    | `<json>{"a":1}</json>` | False | True |
    | `Sure:\n```json\n{"a":1}\n```\nDone` | False | True |
    | `Here: {"a": 1} ok` | False | True |
    | two fenced blocks | False | False |
    | `{"a":1}` | True | True |
    | `{"b":1}` | False | False |
    | `<json>{"b":1}</json>` | False | False |

  - Value checks apply to the extracted value: with `values {"a": 2}`, `<json>{"a":1}</json>` is
    not extractable.
  - Other graders leave `extractable` as `None`.
- [ ] **Step 2: Write the failing tests** in `tests/test_quality.py`.
  - The summary of a `json_schema` suite has `extractable_accuracy == 0.5` for records with
    `extractable` [True, False].
  - `extractable_ci95` equals `wilson(1, 2)`.
  - Non-JSON suites don't have these keys.
  - A record whose grade became `"error"` or `"truncated"` has `extractable = False`.
- [ ] **Step 3: Implement.**
  - **Candidate texts**, tried in this order, stopping at the first that parses and passes:
    1. If there is exactly one fenced block, its body.
    2. If there is exactly one `<json>…</json>`, its body.
    3. The balanced span from the first `{` or `[`. Track string literals and escapes so that
       brackets inside strings don't count.

    More than one fence (or more than one tag) disqualifies that source only.
  - **Parsing and checks.** Use the same parse options (`parse_constant`, `parse_float`) and the
    same schema and `values` checks as the strict path. Refactor those checks into one helper used
    by both paths.
  - **Recording.** `quality.py` records `g.extractable`, or `False` on error, truncation or a grader
    exception.
  - **Version.** Change `bundle_format` to `"m2"` in `runner.py`, and update any test that asserts
    `"m1"`.
- [ ] **Step 4: Run the full suite and ruff.** Expected: everything passes.
- [ ] **Step 5: Commit**: `"Lenient 'extractable' JSON score alongside strict; bundle_format m2"`

### Task 3: Deterministic resampling

**Files:**
- Create: `src/bakeoff/resample.py`
- Test: `tests/test_resample.py`

**Interfaces:**
- Produces:
  - `class SplitMix64(seed: int)` with `.next() -> int`, returning 64-bit unsigned values.
  - `verdict_seed(week: str, tier: str, track: str, cand_sha: str, pick_sha: str) -> int`
  - `paired_diff_ci(suites: list[tuple[list[bool], list[bool]]], resamples: int, ci: float, seed: int) -> tuple[float, float, float]`
    returns `(diff, lo, hi)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_splitmix64_reference_vector`: the first 5 outputs for seed `1234567` are
    `6457827717110365317, 3203168211198807973, 9817491932198370423, 4593380528125082431, 16408922859458223821`.
    This is the published reference vector.
    - If your implementation of the algorithm below doesn't produce it, compile Vigna's ten-line
      reference C (`splitmix64.c`) with `cc` in the scratchpad, and compare the outputs.
    - If the reference agrees with you and not with the vector, report DONE_WITH_CONCERNS with both
      sequences, and pin the reference output.
    - Never pin output that hasn't been checked independently.
  - `test_verdict_seed_derivation`: equals
    `int.from_bytes(sha256(b"bakeoff-verdict-v1|2026-10-19|24gb|general|" + b"a"*64 + b"|" + b"b"*64).digest()[:8], "big")`.
  - `test_identical_pairs_zero`: identical candidate and pick data give `(0.0, 0.0, 0.0)`.
  - `test_ci_golden`:
    - Input: one suite, candidate `[True]*80 + [False]*20` and pick `[True]*70 + [False]*30`,
      paired in that order; `resamples=10000`, `ci=0.95`, `seed=42`.
    - `diff == 0.1`.
    - Pin the `(lo, hi)` your implementation produces as a golden value, computed once and pasted
      into the test.
    - The hand-off must show that `lo < 0.1 < hi`, and that both are within ±0.03 of the analytic
      check in the next test.
  - `test_bootstrap_matches_analytic`: on 5 seeded synthetic datasets with n=200, `lo` and `hi`
    are each within 0.015 of `d ± 1.96·sqrt((b + c) − (b − c)²/n)/n`. Here `b` and `c` are the
    discordant counts (candidate passes and pick fails, and the reverse). This is the independent
    cross-check.
  - `test_multi_suite_mean`: for two suites with diffs 0.2 and 0.0, `diff == 0.1`.
- [ ] **Step 2: Run them.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
  - **splitmix64.** All arithmetic is mod 2⁶⁴:
    ```
    state += 0x9E3779B97F4A7C15
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)
    ```
  - **`paired_diff_ci`:**
    - The caller passes suites already ordered by suite key, with items paired and ordered by id.
    - Precompute `d_i = int(c_i) - int(p_i)` for each suite.
    - Each of the `resamples` iterations does this for every suite, in order:
      1. draw `n` indices `rng.next() % n`, one after another;
      2. compute `sum(d[idx]) / n`.

      The iteration's statistic is the mean of those per-suite values, summed in suite order and
      divided by the suite count.
    - Sort the statistics. The CI is `s[floor((1-ci)/2·B)]` to `s[ceil((1+ci)/2·B) − 1]`.
    - `diff` is the observed mean of the per-suite mean differences.
    - Document in the module docstring that `next() % n` has a negligible modulo bias for n < 2³².
- [ ] **Step 4: Run the tests and ruff.** Expected: PASS. Note the runtime of `test_ci_golden`; it
  should be under 2 s.
- [ ] **Step 5: Commit**: `"Deterministic paired bootstrap: splitmix64, seed derivation, CI"`

### Task 4: Input loaders

**Files:**
- Create: `src/bakeoff/weekcfg.py`, `tiers.toml` (the spec's three tiers, with the 12 GB tier as
  `"1× RTX 3060 12 GB"`), `site.toml` (the spec's block, `launched = false`, empty newsletter
  `action`)
- Test: `tests/test_weekcfg.py`

**Interfaces:**
- Produces:
  - `class WeekInputError(ValueError)`
  - `Tier(id, name, envelope)`, frozen. `load_tiers(path) -> dict[str, Tier]`, keeping file order.
  - `RunEntry(bundle: Path, tier: str, license: str, source_url: str, license_url: str | None)`
  - `Override(tier, track, run, verdict, reason)`
  - `WeekInput(week: str, published_at: str, runs: list[RunEntry], judges: list[Path], overrides: list[Override], correction: str | None, source_bytes: bytes)`
  - `load_week(path) -> WeekInput`. Bundle paths are resolved relative to the file.
  - `SiteConfig(launched: bool, base_url: str, repo_url: str, newsletter_action: str, email_field: str)`
    and `load_site(path) -> SiteConfig`.

- [ ] **Step 1: Write the failing tests.** Each rejection raises `WeekInputError` with a message
  that names the field.
  - The shipped `tiers.toml` and `site.toml` load.
  - These are rejected:
    - an unknown key at any level;
    - a missing `week`;
    - `week = "2026-10-20"` (a Tuesday);
    - `published_at` not matching `YYYY-MM-DDTHH:MM:SSZ`;
    - a run whose tier isn't in `tiers.toml` (`load_week` takes the tiers dict);
    - `source_url = "javascript:alert(1)"`;
    - `source_url = "http://example.com"`;
    - `license = "other"` without `license_url`;
    - an empty license;
    - an override with an empty `reason`;
    - an override verdict outside `replace`, `hold` or `watch`;
    - an override naming a `run` (run_id = the bundle directory's basename) that isn't among the
      `[[run]]` entries;
    - a duplicate tier id or envelope in `tiers.toml`;
    - a tier id that isn't a slug;
    - `site.toml` with a non-https `base_url`;
    - a non-empty newsletter `action` that isn't https.
  - `source_bytes` equals the file's bytes.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** with `tomllib`. Use plain validation functions in the style of M1's
  `config.py`, failing closed.
- [ ] **Step 4: Run the tests and ruff.** Expected: PASS.
- [ ] **Step 5: Commit**: `"Loaders for tiers.toml, week.toml and site.toml"`

### Task 5: Run view and eligibility

**Files:**
- Create: `src/bakeoff/runview.py`
- Test: `tests/test_runview.py`, plus the fixture builder `tests/weekfix.py`, which later tasks
  reuse.

**Interfaces:**
- Consumes: `bundle.write_bundle`, for fixtures only.
- Produces:
  - `class RunViewError(ValueError)`
  - `RunView` (a dataclass) with these fields:
    - identity: `run_id`, `bundle_sha256`, `path`, `bundle_format`, `status`, `envelope`;
    - `model: dict` (`label`, `weights`, `quant`, `runtime`);
    - quality: `tracks: dict[str, float | None]`, `suites: dict[str, dict]` (suite key → summary),
      `items: dict[str, dict[str, bool]]` (suite key → item id → passed);
    - speed and fit: `short_decode: float | None`, `max_context: int | None`,
      `speed: dict`, `fit: dict`;
    - memory and provenance: `size_vram: int | None`, `harness: dict` (`version`, `git_commit`);
    - `meta: dict`.
  - `load_run(path: Path) -> RunView`
  - `model_key(run) -> tuple[str, str, str]`, which is `(weights, quant, runtime)`.
  - `ineligible_reasons(run, rules: dict) -> list[str]`. The reasons, in this order:
    `status_incomplete`, `no_fit`, `context_below_min`, `no_speed`, `decode_below_min`.
  - `tests/weekfix.py`: `make_bundle(dir, *, label, weights, quant="Q4_K_M", runtime="ollama", envelope, grades: dict[str, list[bool]], suite_hashes: dict[str, str] | None = None, decode=40.0, max_context=32768, status="complete", fmt="m2", speed=True, fit=True) -> Path`
    - It writes a real bundle through `write_bundle`.
    - `grades` maps a suite key such as `"general/gsm8k"` to per-item results, with ids `"<name>-<i>"`.

- [ ] **Step 1: Write the failing tests.**
  - `load_run` on a fixture reads the track score and the items map.
  - `short_decode` comes from `speed["short@1"]["decode_tps"]["median"]`, and `max_context` from
    `fit["max_context_passed"]`.
  - A bundle built with `speed=False, fit=False` loads, and its reasons are `["no_fit", "no_speed"]`.
  - `status="incomplete"` gives `status_incomplete` first.
  - With rules v1, a bundle with `decode=9.9` gives `decode_below_min`, and `max_context=8192`
    gives `context_below_min`.
  - Private-suite records are read from `private/`.
  - A judge suite (accuracy `None`) is left out of `items`.
  - An `m1` bundle loads, and its suites have no `extractable_accuracy`.
  - A missing or corrupt `bundle.json` raises `RunViewError`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** The record file for suite `track/name` is
  `quality/{track}.{name}.jsonl` or `private/{track}.{name}.jsonl`. An item counts as passed only
  when `pass is True`.
- [ ] **Step 4: Run the tests and ruff.** Expected: PASS.
- [ ] **Step 5: Commit**: `"Run view over bundles; eligibility reasons"`

### Task 6: Verdict rules

**Files:**
- Create: `src/bakeoff/verdicts.py`
- Test: `tests/test_verdicts.py` (built with `tests/weekfix.py` and real `load_run`)

**Interfaces:**
- Consumes: `RunView`, `model_key`, `ineligible_reasons`, `paired_diff_ci`, `verdict_seed`,
  `Override`.
- Produces:
  - `RULES_V1 = {"version": 1, "min_context": 16384, "min_decode_tps": 10.0, "speed_ratio": 0.8, "context_cap": 32768, "resamples": 10000, "ci": 0.95}`
  - `class VerdictError(ValueError)`
  - `Verdict` (a dataclass) with these fields:
    - `computed`, `final`: one of `replace`, `hold`, `watch`, `initial_pick`;
    - `override_reason: str | None`;
    - `diff: float | None`, `diff_ci95: list[float] | None`;
    - `reasons: list[str]`;
    - `outranked_by: str | None`;
    - `new_pick: bool`.
  - `TierResult` (a dataclass) with these fields:
    - `ineligible: dict[str, list[str]]`;
    - `pick_tracks: dict[str, list[str]]`, mapping run_id to the tracks where that run is the
      pick's control;
    - `verdicts: dict[str, dict[str, Verdict]]`;
    - `pick_status: dict[str, str]`, with values `re-measured`, `not_re-measured` or `none`;
    - `picks_after: dict[str, dict]`.
  - `assess_tier(week: str, tier: str, runs: list[RunView], picks_before: dict[str, dict], overrides: list[Override], rules: dict) -> TierResult`
  - A pick dict has: `tier, track, label, weights, quant, runtime, since_week, bundle_sha256, run_id`.
  - The reason codes are exactly:
    - `ci_includes_zero`, `not_better`;
    - `slower_than_ratio`, `less_context`;
    - `pick_not_remeasured`, `suite_set_mismatch`, `suite_hash_mismatch`, `item_ids_mismatch`.

Follow spec rules 1–10, plus the plan rulings below.

- [ ] **Step 1: Write the failing table-driven tests,** using `resamples=2000` through a rules copy
  for speed.
  - **Clear win → replace.**
    - Setup: candidate 90/100 against pick 60/100 on one general suite; similar decode; equal
      context.
    - Result: `replace` with `new_pick`, and `picks_after["general"]["run_id"]` is the candidate's.
  - **Small edge → watch.** A 62 vs 60 result is `watch` with `ci_includes_zero`.
  - **Worse → hold.** 55 vs 60 is `hold` with `not_better`.
  - **Speed and context failures.**
    - A clear win with decode at 0.5× the pick's is `watch` with `slower_than_ratio`.
    - A clear win with `max_context` 16384 against the pick's 32768 is `watch` with `less_context`.
    - A pick at 65536 against a candidate at 32768 counts as enough context, because of the 32768
      cap.
  - **Comparability failures.**
    - A suite-hash mismatch gives `watch`, with `suite_hash_mismatch`, and `diff = None`.
    - A missing suite gives `suite_set_mismatch`.
    - Missing pick control gives `pick_status "not_re-measured"`. Every candidate is capped at
      `watch` with `pick_not_remeasured`, and `picks_after` equals `picks_before`.
  - **No pick (first week).**
    - With 3 eligible runs, the highest score becomes `initial_pick`. The others are compared
      against it and get `hold`.
    - Tie-break: higher decode, then the lower run_id.
  - **Two clear wins.** The higher score becomes the new pick. The other keeps `replace` and gets
    `outranked_by`.
  - **Overrides.**
    - `hold → replace` makes the run the new pick, keeps `computed = "hold"` and sets
      `override_reason`.
    - `replace → hold` removes the run from pick selection.
    - These raise `VerdictError`:
      - an override on an ineligible run;
      - an override on the pick's own control run;
      - an override on a track the run has no score in;
      - an override in a tier × track with no current pick (plan ruling: the initial pick is always
        computed).
  - **Eligibility.** Ineligible runs appear only in `ineligible`.
  - **Determinism.** Assessing twice gives equal results.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement `assess_tier`.**
  - **Comparison inputs.**
    - Pair items by id, sorted in byte order.
    - Suites are sorted by key, and only suites with non-null accuracy are compared.
    - The seed comes from `verdict_seed(week, tier, track, cand.bundle_sha256, pick.bundle_sha256)`.
  - **The pick's control run.**
    - It is the eligible run whose `model_key` equals the pick's.
    - If there are several, use the lowest run_id.
    - If the control is ineligible, the pick counts as `not_re-measured`.
  - **The pick's `since_week`** carries over while the pick is unchanged. Its `bundle_sha256` and
    `run_id` update to this week's control run.
- [ ] **Step 4: Run the tests and ruff.** Expected: PASS.
- [ ] **Step 5: Commit**: `"Verdict rules v1: eligibility, paired comparison, picks, overrides"`

### Task 7: Week document and schema

**Files:**
- Create: `src/bakeoff/week.py`, `src/bakeoff/schemas/week.v1.json`
- Modify: `src/bakeoff/graders.py` (the schema subset)
- Test: `tests/test_week.py`, `tests/test_graders.py`

**Interfaces:**
- Consumes: `WeekInput`, `RunEntry`, `Tier`, `RunView`, `TierResult`, `RULES_V1`.
- Produces:
  - `build_week(inp: WeekInput, tiers: dict[str, Tier], runs: dict[str, tuple[RunView, RunEntry]], results: dict[str, TierResult], judges: list[dict], *, revision: int, published_at: str, corrections: list[dict], rules: dict) -> dict`
  - `load_week_schema() -> dict`, which reads the packaged file through `importlib.resources`.
  - `validate_week(doc) -> list[str]`.
  - The week document follows the spec's `week.json` shape, with three plan rulings:
    1. Each run also has `pick_tracks: [track]`.
    2. The week has `picks_after: {tier: {track: pick}}`. That is the pick history Task 8 derives
       from.
    3. `memory` contains only `size_vram`, and only when the bundle reports it. System-wide peak
       memory isn't specific to the model, so it's left out.
  - `bundle_path` is `results/<week>/runs/<run_id>`.
  - `judge` entries are `{a, b, wins, ties, losses, errors, judge_model, bundle_sha256}`, where `a`
    and `b` are run_ids mapped from the judge bundle's hashes.

Plan ruling on the schema subset: the spec keeps the schema inside M1's checker, but describing
maps and nullable fields needs three small extensions. They still fail closed, and no new dependency
comes with them.
- `type` may be a list of type names.
- `additionalProperties` may be a schema object, which applies to keys not in `properties`.
- The annotation keywords `$schema`, `$id`, `title` and `description` are accepted and ignored.

- [ ] **Step 1: Write the failing tests.**
  - **Checker:** `type: ["number","null"]` accepts both and rejects a string. A map-valued
    `additionalProperties` validates each value. Annotations are accepted. An unknown keyword is
    still an error. The existing grader tests stay green.
  - **Week document:**
    - For a 2-tier fixture week, `build_week` output passes `validate_week`.
    - `schema == "bakeoff.week.v1"`.
    - The rules are copied in.
    - `harness` lists distinct `{version, git_commit}` values, sorted.
    - Floats are written unrounded.
    - A doc with a wrong verdict enum fails validation.
  - **Schema strictness:** `additionalProperties: false` applies on every object with fixed keys.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Write `week.v1.json` by hand. Use `$id` set to
  `https://bakeoff.si/api/v1/schema/week.v1.json`, and include `description` strings on the main
  fields.
- [ ] **Step 4: Run the full suite and ruff.** Expected: PASS.
- [ ] **Step 5: Commit**: `"Week document v1 and its JSON Schema"`

### Task 8: Publish

**Files:**
- Create: `src/bakeoff/publish.py`
- Test: `tests/test_publish.py`

**Interfaces:**
- Consumes: Tasks 1 and 4–7.
- Produces:
  - `class PublishError(Exception)`
  - `publish(week_toml: Path, root: Path, *, username: str | None, denylist: Path | None, username_skip_output: bool = False) -> Path`,
    which returns `root/results/<week>`.
  - `week_docs(root) -> list[dict]`: every `results/*/week.json`, sorted by week.
  - `write_picks(root) -> None`: `results/picks.json` becomes the latest week's `picks_after`, or
    `{}`.

- [ ] **Step 1: Write the failing tests.** Fixtures go in a `tmp_path` root that holds a copy of
  `tiers.toml` and `site.toml`.
  - **Happy path.** Publishing a 2-run week creates:
    - `results/<week>/week.toml`, byte-identical to the input;
    - `week.json`, which validates;
    - `runs/<run_id>/`, with bundles that verify and `bundle_sha256` unchanged;
    - `results/picks.json`.
  - **Reproducibility.** Publishing the same input into two fresh roots gives byte-identical
    `results/` trees.
  - **Refusals.** Each case below raises `PublishError` and leaves no `results/<week>` and no
    staging directory:
    - a corrupted bundle file;
    - a planted scrub finding, such as `/home/alice` in a model label (`scrub-ok`);
    - an absolute local path in `week.toml`;
    - a duplicate run_id;
    - an envelope that doesn't match the tier;
    - `status` neither `complete` nor `incomplete`;
    - a judge bundle whose `a` or `b` isn't a run in the week;
    - an existing week without `[correction]`;
    - a `[correction]` for a week that doesn't exist.
  - **Week 2.** It derives `picks_before` from week 1's `picks_after`.
  - **Corrections.**
    - Re-publishing week 1 with `[correction]` gives `revision: 2`. `corrections` gets
      `{revision: 2, published_at: <new toml value>, note}`, and the original `published_at` is
      kept.
    - The same correction is refused once week 2 exists, if it changes week 1's `picks_after`.
    - A correction may reuse an already-published bundle path. A same-run_id bundle with a different
      `bundle_sha256` is refused, because published bundles are immutable.
  - **Interrupted publish.** Monkeypatch the copy step to raise `KeyboardInterrupt` partway
    through. The staging directory is gone, and a rerun succeeds.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
  1. Load the inputs.
  2. Verify every run and judge bundle in place with `verify_bundle`.
  3. Load each run with `load_run`, and group the runs by tier.
  4. `picks_before` comes from the previous week's `picks_after`.
  5. Call `assess_tier` for each tier.
  6. Build the week document with `build_week`, then check it with `validate_week`.
  7. Stage everything into `results/.<week>.staging/`:
     - copy each bundle with `shutil.copytree`;
     - re-verify each copy;
     - write `week.toml` and `week.json`;
     - run `scan_text` over the staged `week.toml` and `week.json`.
  8. Replace or rename the staging directory into `results/<week>`, then call `write_picks`.
  9. Clean up the staging directory in a `finally` block.

  A correction copies the existing bundles over from the old `results/<week>/runs/`.
- [ ] **Step 4: Run the full suite and ruff.** Expected: PASS.
- [ ] **Step 5: Commit**: `"bakeoff publish: verify, assess, stage, write results and picks"`

### Task 9: Pages and formatting

**Files:**
- Create: `src/bakeoff/web/__init__.py`, `src/bakeoff/web/fmt.py`, `src/bakeoff/web/pages.py`,
  `site/style.css`, `site_src/method.html`, `site_src/privacy.html`
- Test: `tests/test_web_pages.py`, `tests/htmlcheck.py`

**Interfaces:**
- Consumes: `SiteConfig`; week documents shaped as in Task 7.
- Produces:
  - **Formatting** (`fmt.py`; every function returns `"—"` for `None`):
    - `pct(x) -> "87.0%"`
    - `pm(ci) -> "±6.9"` (half-width in points, one decimal place)
    - `tps(x) -> "47.7"`
    - `ms(seconds) -> "42 ms"`
    - `kctx(n) -> "32K"`
  - **Pages** (`pages.py`):
    - `holding_page(site) -> str`
    - `week_page(doc, site) -> str`
    - `archive_page(index: list[dict], site) -> str`
    - `wrap_static(fragment: str, title: str, site) -> str`
    - `newsletter_form(site) -> str`, which returns `""` when `newsletter_action` is empty.

- [ ] **Step 1: Write the failing tests.**
  - **`fmt`:** golden values, plus `None` → `"—"`.
  - **Holding page:** it keeps the current page's copy (lede, the three bullets, "First results
    soon."). It links `/style.css`, and has no `<style>`, no `style=` and no `<script>`.
  - **Newsletter form:**
    - It appears only when `action` is set.
    - Its `method` is `post` and its `action` is the escaped URL.
    - The input name is `email_field`, and it links to `/privacy/`.
  - **Week page:**
    - A model label `Qwen3 "Instruct" <beta> & co` comes out escaped in both text and `title`
      attributes.
    - The page has a section per tier, the pick row, verdict text, and the override line
      ("Computed: hold, owner: replace", then the reason).
    - It shows "Tested, doesn't fit" with the reasons, strict and extractable side by side for
      structured JSON, "not measured" for m1 bundles, and the raw-bundle link
      `{repo_url}/tree/main/{bundle_path}`.
  - **Judge results:** they appear as information only ("Judge: 12 wins, 3 ties, 5 losses vs …").
  - **Well-formed HTML:** every page passes `tests/htmlcheck.py::assert_balanced(html)`, a small
    `html.parser.HTMLParser` subclass that checks tag balance (void elements excepted). Create it in
    this task; Tasks 10 and 11 reuse it.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Stylesheet.** Move the current `site/index.html` `<style>` rules into `site/style.css`, and
    add table styles:
    - tables scroll horizontally inside a wrapper at phone width;
    - numbers are right-aligned;
    - verdict badges use the existing palette (`--blue` for replace, `--muted` for hold, an amber
      added as a token for watch).
  - **Shared layout.** All pages share it: the logo header, `<main>`, and a footer with links to
    `/method/`, the week JSON, `/feed.xml` (when launched) and the form.
  - **Method page.** `site_src/method.html` is a body fragment covering:
    - the tiers;
    - rules v1 with their thresholds;
    - the suites and their licenses;
    - "reproduce a week" (`uv sync`, then `bakeoff publish`).
  - **Privacy page.** `site_src/privacy.html` says what the sign-up collects (the email address
    only), who stores it (the newsletter provider), double opt-in, how to unsubscribe, and that
    there are no analytics or cookies.
- [ ] **Step 4: Run the tests and ruff.** Expected: PASS.
- [ ] **Step 5: Commit**: `"Static pages, display formatting, stylesheet, method and privacy"`

### Task 10: Site build, API, feed, headers and CLI

**Files:**
- Create: `src/bakeoff/web/build.py`
- Modify: `src/bakeoff/cli.py`, `site/index.html` (now generator output)
- Test: `tests/test_web_build.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 8 (`publish`, `week_docs`) and Task 9.
- Produces:
  - `OWNED = ("index.html", "weeks", "api", "feed.xml", "_headers", "method", "privacy")`
  - `render_site(root: Path, *, launched: bool | None = None) -> dict[str, bytes]`. It is pure and
    reads `site.toml`, `site_src/` and `week_docs(root)`.
  - `write_site(files, out_dir) -> None`. It deletes stale files under `OWNED` paths first.
  - `check_site(root) -> list[str]`. Each problem is reported as `changed: <path>`,
    `missing: <path>` or `stray: <path>`.
  - `atom_feed(docs, site) -> str`
  - `headers_file(site) -> str`
  - **CLI:**
    - `bakeoff publish <week.toml> [--root .] [--denylist F] [--no-username]` runs `publish`, then
      `write_site(render_site(root), root/"site")`.
    - `bakeoff site [--root .] [--check | --preview DIR]`.
    - The exit codes are 0 for ok, 1 for problems and 2 for usage errors.

- [ ] **Step 1: Write the failing tests.**
  - **Launch gate:**
    - With `launched=false`, `render_site` keys are exactly `index.html`, `_headers`,
      `method/index.html` and `privacy/index.html`. `index.html` is the holding page, and there is
      no `weeks/`, `api/` or `feed.xml`.
    - With `launched=true` and two published fixture weeks, the keys include:
      - `index.html` (the latest week);
      - `weeks/index.html` and `weeks/<w>/index.html`;
      - `api/v1/latest.json`, `api/v1/weeks/index.json`, `api/v1/weeks/<w>.json` and
        `api/v1/picks.json`;
      - `api/v1/schema/week.v1.json` and `feed.xml`.
  - **API:**
    - `latest.json` equals the newest `week.json` byte for byte.
    - `weeks/index.json` is newest first, with `url = base_url + "/api/v1/weeks/<w>.json"`.
  - **Feed:**
    - `feed.xml` parses with `xml.etree.ElementTree`.
    - Entry ids are `tag:bakeoff.si,2026:week/<w>`, and `updated` is the latest revision's
      `published_at`.
    - The hostile label is escaped.
  - **Headers:**
    - `_headers` has the spec's CSP under `/*`.
    - `form-action` is `'none'` without a form, and is the action's origin with one.
    - `/api/v1/*` gets `Access-Control-Allow-Origin: *`.
  - **`--check`:** it is clean right after `write_site`. It reports `changed:` after a hand edit
    and `stray:` for a new file under `weeks/`.
  - **Determinism:** two renders are byte-identical.
  - **CLI:**
    - `site --preview <tmp>` writes a launched build plus the hand-maintained assets.
    - `publish` end-to-end on fixtures leaves `site --check` clean.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement,** then run `uv run --project <repo> bakeoff site --root <repo>`.
  - That regenerates the committed `site/index.html`, `_headers`, `method/` and `privacy/`.
  - Check by eye that `site/index.html` still reads like today's holding page.
  - `--check` must then be clean.
- [ ] **Step 4: Run the full suite and ruff.** Expected: PASS.
- [ ] **Step 5: Commit**: `"Site build: API, feed, headers, launch gate, check, preview; publish and site commands"`

### Task 11: CI, docs and the M2 check

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`, `PLAN.md`, `.gitignore`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the CI workflow.**
  - It runs on `pull_request` and on `push` to `main`, with `permissions: contents: read`, on
    `ubuntu-latest`.
  - **Steps:**
    1. Checkout.
    2. `astral-sh/setup-uv`.
    3. `uv sync --locked`.
    4. `uv run ruff check`, then `uv run ruff format --check`.
    5. `uv run pytest -q`.
    6. `uv run bakeoff site --check`.
    7. Run `uv run bakeoff verify --no-username` for every directory matching
       `results/*/runs/*/` and `results/*/judge/*/`. Use a shell loop that tolerates zero matches.
    8. `uv run python scripts/scan_repo.py`.
  - **Pinning.** Pin both actions by full commit SHA, with the tag in a comment. Resolve the SHAs
    with `mise exec gh@2.91.0 -- gh api repos/<owner>/<repo>/commits/<tag> --jq .sha` for the
    latest `v4` of `actions/checkout` and the latest major of `astral-sh/setup-uv`.
  - Validate the YAML with `python -c "import yaml"` if it's available; otherwise read it carefully.
- [ ] **Step 2: Cross-check with real data.** Put everything under `runs/` (git-ignored).
  1. Make a scratch root, `runs/m2-preview-root/`. Copy in `tiers.toml`, `site.toml`, `site_src/`
     and `site/`.
  2. Write `runs/m2-preview-root/week.toml`:
     - week `2026-10-19`;
     - one `[[run]]` pointing at the Granite bundle under `runs/final/` (relative path), tier
       `36gb-unified`;
     - license `Apache-2.0`;
     - `source_url = "https://huggingface.co/ibm-granite/granite-3.3-8b-instruct"`.
  3. Run `bakeoff publish runs/m2-preview-root/week.toml --root runs/m2-preview-root`, then
     `bakeoff site --root runs/m2-preview-root --preview runs/m2-preview`.
  4. **Expected:**
     - Exit 0.
     - Granite is the `initial_pick` on every track it has.
     - The week page numbers match `bundle.json` after rounding: gsm8k 87.0%, short@1 decode ≈ 47.7
       and 32K context.
     - Structured shows strict 40.0% next to an extractable figure. That bundle is `m1`, so the
       figure reads "not measured". Note it in the hand-off.
  5. Validate every generated page:
     - each HTML page passes `tests/htmlcheck.py::assert_balanced`;
     - `feed.xml` parses;
     - every API JSON file validates against the schema.

     Report the counts.
  6. Report the command for the owner to view the preview locally:
     `python3 -m http.server -d runs/m2-preview 8000`.
- [ ] **Step 3: Update the docs.**
  - **README:** add a "Publishing a week" section covering `week.toml`, `bakeoff publish`,
    `bakeoff site --check` and `--preview`, the launch gate and the API paths. Add rows for
    `tiers.toml`, `site.toml`, `site_src/` and `results/` in the layout table.
  - **PLAN.md:**
    - Mark M2 ✅.
    - Resolve the two open questions: "results to site" is decided as generate and commit, and
      "schema versioning" as `/api/v1` additive-only.
    - Under "Repo admin to finish", add an owner checklist for the first Git deploy, with the 5
      checks below.
  - **`.gitignore`:** no change if `runs/` already covers the previews.
- [ ] **Step 4: Final verification.**
  - Run `pytest -q`, `ruff check`, `ruff format --check`, `bakeoff site --check` and
    `scripts/scan_repo.py --username "$(whoami)"`, adding private terms on the command line only.
  - **Expected:** all pass, and there are 0 findings.
- [ ] **Step 5: Commit**: `"CI workflow; publishing docs; mark M2 done"`

**First Git deploy checks** (for the owner, after the merge; listed in PLAN.md):
1. bakeoff.si and www resolve, and www redirects with a 301 to the root.
2. workers.dev is still off.
3. `curl -sI https://bakeoff.si/` shows the CSP header.
4. `https://bakeoff.si/_headers` and `https://bakeoff.si/results/` return 404.
5. The holding page looks unchanged.
