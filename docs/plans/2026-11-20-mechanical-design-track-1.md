# Mechanical design track, part 1: rounds, sandbox, part checks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run edit-a-part (K2) and fit-a-part (K3) tasks end to end: fixed feedback rounds against any endpoint, model CadQuery executed in a sandbox, machine checks, and the results in the run bundle as information only.

**Architecture:** A new `bakeoff.mech` package. Model-written CadQuery runs only inside a sandboxed subprocess (`bakeoff.mech._exec`) that returns plain JSON *facts* (measurements). Checks are pure functions of those facts, so they're testable without CadQuery and re-gradable from a bundle. A round driver runs the chat loop. The runner gains a `mech` pass between `quality` and `speed`. Results go to a new bundle section `mechanical`, never into `quality.tracks`, so `verdicts.py` can't make picks from them.

**Tech Stack:** Python ≥3.11, httpx (core), CadQuery (optional extra `mech`), bubblewrap (`bwrap`) on Linux lab hosts, pytest, ruff.

**Spec:** `docs/specs/2026-11-20-mechanical-design-track-design.md`

**Later plans (not this one):**
- **Part 2:** K5 system tasks: the catalog, wiring and consistency checks, temperature zones, the controller simulator and the safety gate.
- **Part 3:** the calibration anchor; the site's information-only display; enabling the pass in weekend runs (an athena change), plus host setup (`bwrap` and the extra on the Linux lab hosts, done by the owner).

## Global Constraints

- The core install keeps one dependency (`httpx`). CadQuery comes only through `[project.optional-dependencies] mech`. Pin the newest CadQuery release that is at least 14 days old and has wheels for Python 3.12 on linux x86_64 and macOS arm64.
- Model-written code runs only through `bakeoff.mech.sandbox`. It never runs via `exec` in the harness process.
- Lab runs use the `bwrap` backend. `subprocess` (rlimits, no isolation) is for development and CI only: it needs the env var `BAKEOFF_MECH_UNSANDBOXED=1`, is recorded in the bundle, and `publish` refuses mechanical results produced with it.
- Mechanical results never enter `quality.suites` or `quality.tracks`. The bundle's `mechanical.info_only` is `true`.
- v1 is public tasks only: loading a mechanical suite with `split = "private"` is an error.
- Rounds: `rounds = 3` by default (one reply plus two revisions). `max_tokens` applies per reply. Each request carries the whole conversation (system, brief, earlier replies' content without reasoning, earlier reports). Requests use `temperature=0.0, seed=0`.
- Parts are CadQuery only. A part block must leave a module-level `result` (a `cq.Workplane` or `cq.Shape`) that is exactly one valid solid.
- All geometry is in one assembly frame, in millimetres. There are no placement transforms in v1.
- The public repo scrub rules apply (no hostnames, IPs, users or local paths in code, tests or bundles); ruff line length is 100.

## Review Focus

1. **Hostile part code** (an infinite loop, a 50 GB allocation, `os.fork` bombs) ends as a failed `builds` check with reason `timeout` or `memory`, and the run continues. Test in Task 3.
2. **Part code reading host secrets or the network** (`open(os.path.expanduser("~/.ssh/id_ed25519"))`, `urllib.request.urlopen(...)`) fails inside `bwrap`: no home directory is bound and the network is unshared. Test in Task 3 (bwrap-only, skipped where `bwrap` is missing).
3. **Malformed replies**: the same part twice, a part block with no `result`, a `result` holding two solids, an unclosed fence, or a reply that's only reasoning. Each gives a clear format or `builds` failure with a reason the model can act on. Tests in Tasks 2 and 3.
4. **Reasoning models**: only `content` is parsed and fed back into the history, never the reasoning. Test in Task 6.
5. **History growth on llama.cpp**: a later round whose worst case exceeds the server's shared context is recorded as a round error `context`, not a crash, and the task fails. Test in Task 6.

---

## File Structure

- `src/bakeoff/mech/__init__.py`: package marker; `MECH_CONCURRENCY = 4`, `SANDBOX_CONCURRENCY = 2`.
- `src/bakeoff/mech/tasks.py`: rounds-suite manifest and item loading, prompt construction.
- `src/bakeoff/mech/reply.py`: parse a reply into named part sources.
- `src/bakeoff/mech/sandbox.py`: run a job in `bwrap` or `subprocess` with limits; `FakeSandbox` for tests.
- `src/bakeoff/mech/_exec.py`: the sandboxed entry point (stdin JSON → stdout JSON). It imports CadQuery and nothing from the harness except `measure`.
- `src/bakeoff/mech/measure.py`: geometry measurements (CadQuery/OCP), called only from `_exec`.
- `src/bakeoff/mech/checks.py`: the check registry, pure functions over facts.
- `src/bakeoff/mech/report.py`: the feedback message text.
- `src/bakeoff/mech/rounds.py`: the round driver and the pass runner.
- `src/bakeoff/config.py`, `runner.py`, `cli.py`, `publish.py`, `suites.py`, `scripts/lock_suites.py`: integration.
- `tests/test_mech_*.py`, `tests/data/mech/`: fixtures (two tiny tasks with reference replies).
- `docs/mechanical-authoring.md`: how the owner writes a task.

---

### Task 1: Rounds-suite format and loader

**Files:**
- Create: `src/bakeoff/mech/__init__.py`, `src/bakeoff/mech/tasks.py`
- Modify: `src/bakeoff/suites.py` (`load_suite`), `scripts/lock_suites.py`
- Create: `tests/test_mech_tasks.py`, `tests/data/mech/edit-demo/{manifest.toml,items.jsonl,reference.jsonl}`, `tests/data/mech/fit-demo/{...}`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class MechItem: id: str; kind: str; brief: str; given: dict[str, str]; parts: list[str]; edit_of: str | None; checks: list[dict]`
  - `@dataclass class MechSuite: path: Path; track: str; name: str; version: int; split: str; license: str; rounds: int; max_tokens: int; system: str; items: list[MechItem]; references: dict[str, dict]; hash: str` with property `key -> f"{track}/{name}"`.
  - `load_mech_suite(path: Path) -> MechSuite` (raises `SuiteError`).
  - `first_messages(suite: MechSuite, item: MechItem) -> list[dict]`.
- Manifest: the `suites.py` fields plus `kind = "rounds"` and an optional `rounds` (int, 1..5, default 3). `track` must be `"mechanical"`, there is no `grader`, and `split` must be `"public"`.
- Item line: `{"id", "kind": "edit"|"fit", "brief", "given": {name: source}, "parts": [names the reply must contain], "edit_of": name|null, "checks": [{"check", "part", "params", "visible"}]}`. `edit_of` is required for `edit` and must also appear in `parts`.
- `reference.jsonl` line: `{"id", "good": reply_text, "bad": [reply_text, ...]}`. Every item needs one, with at least one `bad`.
- Hash: `sha256_hex(canonical_json({"manifest", "items", "references"}))`.

- [ ] **Step 1: Write failing tests**: `test_loads_demo_suites` (both fixtures load; `rounds == 3`; `key == "mechanical/edit-demo"`); `test_rejects_private`; `test_rejects_unknown_check` (names come from Task 5's registry; until then, assert against a set `KNOWN_CHECKS` defined in `tasks.py` and replaced by an import in Task 5); `test_edit_needs_edit_of_in_parts`; `test_item_without_reference_rejected`; `test_load_suite_points_rounds_suites_to_mech` (`suites.load_suite` on a rounds suite raises `SuiteError` containing `"bakeoff.mech"`); `test_first_messages_shape` (system then one user message containing the brief and each given part as a fenced block labelled `` ```cadquery part=<name> ``).
- [ ] **Step 2:** `uv run pytest tests/test_mech_tasks.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement.** Reuse `suites.py` helpers (`_require_str`, `_IDENT`, `SPLITS`). In `load_suite`, check `kind` before the grader. `lock_suites.py` locks rounds suites with `load_mech_suite` under the same `{key: {version: hash}}` scheme.
- [ ] **Step 4:** run the tests: PASS, plus `uv run pytest -q tests/test_lock_suites.py tests/test_suites.py`: PASS.
- [ ] **Step 5: Commit** `mech: rounds-suite format and loader`.

### Task 2: Reply parser

**Files:** Create `src/bakeoff/mech/reply.py`, `tests/test_mech_reply.py`

**Interfaces:**
- Produces `@dataclass(frozen=True) class ParsedReply: parts: dict[str, str]; problems: list[str]` and `parse_reply(content: str, expected_parts: list[str]) -> ParsedReply`.
- Grammar: a fence line matching `^```cadquery part=([a-z0-9][a-z0-9_-]*)\s*$`, its body, then a closing line of exactly three backticks. Text outside blocks is ignored, and so are other fenced blocks.
- Problems (exact wording, used verbatim in reports):
  - `"part '<n>' appears more than once"`
  - `"missing part '<n>'"`
  - `"unexpected part '<n>'"`
  - `"unclosed block for part '<n>'"`
  - `"no part blocks found"`

- [ ] **Step 1: Failing tests**: one test per problem; `test_two_parts_ok`; `test_ignores_python_blocks_and_prose`; `test_crlf_and_trailing_spaces`; `test_empty_content` (→ `["no part blocks found"]`). Any problem means the parts are not executed (Task 6).
- [ ] **Step 2:** run: FAIL. **Step 3:** implement. **Step 4:** run: PASS.
- [ ] **Step 5: Commit** `mech: reply parser`.

### Task 3: Sandbox and the sandboxed executor

**Files:** Create `src/bakeoff/mech/sandbox.py`, `src/bakeoff/mech/_exec.py`, `tests/test_mech_sandbox.py`; modify `pyproject.toml` (extra `mech`, pytest marker `kernel`), `uv.lock`.

**Interfaces:**
- `@dataclass(frozen=True) class Limits: wall_s: float = 60.0; mem_mb: int = 4096; cpu_s: int = 60`
- `@dataclass(frozen=True) class SandboxResult: ok: bool; facts: dict | None; error: str | None` (`error` ∈ `"timeout"`, `"memory"`, `"crash: <first 300 chars of stderr>"`, `"bad output"`).
- `class Sandbox(Protocol): backend: str; async def run(self, job: dict) -> SandboxResult`.
- `make_sandbox(backend: str, limits: Limits = Limits()) -> Sandbox`. `"bwrap"` requires the `bwrap` binary. `"subprocess"` requires `BAKEOFF_MECH_UNSANDBOXED=1`. Anything else raises `ValueError`.
- `class FakeSandbox` (in `sandbox.py`, test-only use): takes a callable `job -> SandboxResult` and records jobs.
- Job (stdin to `_exec`): `{"parts": {name: src}, "given": {name: src}, "checks": [check dicts]}`.
- Output (stdout from `_exec`): `{"versions": {"cadquery", "ocp"}, "parts": {name: part_facts}, "given": {name: part_facts}, "checks": [measurement | null, ...]}`, where `part_facts = {"built": bool, "error": str|None, "valid": bool, "solids": int, "bbox": [[x0,y0,z0],[x1,y1,z1]]|None, "volume": float|None, "cylinders": [{"radius", "axis": [x,y,z], "through": [x,y,z]}]}`. The measurements come from Task 4. In this task, `_exec` fills `parts`/`given` and returns `null` for every check.
- bwrap argv: `bwrap --unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind-try /lib /lib --ro-bind-try /lib64 /lib64 --ro-bind-try /etc/ld.so.cache /etc/ld.so.cache --ro-bind <sys.base_prefix> <same> --ro-bind <sys.prefix> <same> --proc /proc --dev /dev --tmpfs /tmp --chdir /tmp <sys.executable> -I -m bakeoff.mech._exec`. No home directory and no other `/etc` are bound. When the harness itself lives outside `sys.prefix` (an editable install), add `--ro-bind <package dir> <same>`.
- Limits via `preexec_fn` with `resource.setrlimit` (`RLIMIT_AS` = mem_mb, `RLIMIT_CPU` = cpu_s, `RLIMIT_NPROC` = 64, `RLIMIT_FSIZE` = 64 MB). Wall clock via `asyncio.wait_for`, which kills the process group.
- Each part's source runs in a fresh dict namespace with `cadquery as cq` preloaded. Exceptions become `built: false, error: "<ExcType>: <msg>"`, truncated to 300 characters.

- [ ] **Step 1: Failing tests:**
  - Without CadQuery (plain pytest, `subprocess` backend with the env var set and a stub `_exec` mode `BAKEOFF_MECH_EXEC_STUB=1` that echoes the job keys): `test_make_sandbox_refuses_subprocess_without_env`, `test_timeout` (stub `sleep`), `test_bad_output`.
  - `@pytest.mark.kernel` (skip unless `import cadquery` works): `test_box_builds` (`result = cq.Workplane().box(10,20,30)` → `volume` 6000 ±1e-6, bbox size `[10,20,30]`), `test_no_result_is_build_failure`, `test_two_solids_counted` (`solids == 2`, `valid` true, and the `builds` check in Task 5 fails it), `test_infinite_loop_times_out`, `test_huge_alloc_is_memory`.
  - `@pytest.mark.kernel` plus skip unless `bwrap` works: `test_bwrap_blocks_home_read` (part code that reads `$HOME/.bashrc` reports `built: false` with `FileNotFoundError`), `test_bwrap_blocks_network` (`socket.create_connection(("1.1.1.1", 53), 2)` fails).
- [ ] **Step 2:** run: FAIL. **Step 3:** implement; add `[project.optional-dependencies] mech = ["cadquery==<pinned>"]` and the `kernel` marker; `uv lock`. **Step 4:** run with and without `uv sync --extra mech`: PASS / kernel tests skipped.
- [ ] **Step 5: Commit** `mech: sandbox and executor`.

### Task 4: Measurements

**Files:** Create `src/bakeoff/mech/measure.py`, `tests/test_mech_measure.py` (all `@pytest.mark.kernel`); modify `_exec.py` to call it.

**Interfaces:** `measure_check(check: dict, parts: dict[str, cq.Shape], given: dict[str, cq.Shape]) -> dict | None`, dispatching on `check["check"]`:

| check | measurement returned |
|---|---|
| `builds`, `bbox`, `volume`, `cylinder` | `None` (from part facts) |
| `probes` | `{"inside": [bool per point]}` (point-in-solid, tolerance 1e-6) |
| `unchanged` | `{"diff_volume": float}`: volume of (new XOR original) after subtracting the `regions` boxes |
| `no_interference` | `{"volumes": {other: float}}`: intersection volume with each named given part |
| `gap` | `{"distance": float}`: minimum distance to `other` (`BRepExtrema_DistShapeShape`) |
| `overhang` | `{"area": float}`: total area of mesh triangles whose downward normal is more than `max_angle_deg` from vertical, excluding faces within 0.01 mm of the part's minimum z (the build plate) |
| `min_wall` | `{"thickness": float}`: minimum, over sample points (face centres plus a 2 mm grid on planar faces), of the distance along the inward normal to the next surface |

A measurement that raises returns `{"error": "<ExcType>: <msg>"}`. Mesh tolerance: linear 0.05 mm, angular 0.1 rad.

- [ ] **Step 1: Failing tests** on known shapes:
  - a 10 mm cube with a Ø4 through-hole: probes inside/outside; `cylinder` facts list radius 2 along z;
  - `unchanged`: the cube against the cube with a 2 mm chamfer on one edge gives `diff_volume` > 0, and 0 within 1e-6 when the chamfered edge's region is excluded;
  - `no_interference`: two overlapping cubes give the right overlap volume;
  - `gap`: cubes 1.5 mm apart give 1.5 ±1e-6;
  - `overhang`: a T-shape gives the underside area of its arms, and a plain cube gives 0;
  - `min_wall`: a 2 mm-walled hollow box gives 2.0 ±0.05.
- [ ] **Step 2:** FAIL. **Step 3:** implement; `_exec` fills `checks[i]`. **Step 4:** PASS.
- [ ] **Step 5: Commit** `mech: geometry measurements`.

### Task 5: Checks and the report

**Files:** Create `src/bakeoff/mech/checks.py`, `src/bakeoff/mech/report.py`, `tests/test_mech_checks.py`; modify `tasks.py` to import the registry instead of `KNOWN_CHECKS`.

**Interfaces:**
- `@dataclass(frozen=True) class CheckResult: check: str; part: str; passed: bool; reason: str; visible: bool`
- `CHECKS: dict[str, Callable[[dict, dict, dict], tuple[bool, str]]]`, each called as `fn(params, part_facts_by_name, measurement)`.
- `evaluate(item: MechItem, facts: dict) -> list[CheckResult]`, in the item's check order.
- Params and pass rules:

| check | params | passes when |
|---|---|---|
| `builds` | `{}` | part built, valid, `solids == 1` |
| `bbox` | `size`, `tol` | each axis within tol |
| `volume` | `value`, `rel_tol` | within rel_tol |
| `cylinder` | `radius`, `tol`, `axis`, `through`, `dist_tol` | some cylinder matches radius, parallel axis (angle < 0.5°), and its axis line passes within `dist_tol` of `through` |
| `probes` | `points: [{at, inside}]` | all match |
| `unchanged` | `original`, `regions: [[min,max]]`, `max_volume` | `diff_volume <= max_volume` |
| `no_interference` | `others`, `max_volume` (default 0.001) | every volume ≤ max |
| `gap` | `other`, `min`, `max` | min ≤ distance ≤ max |
| `overhang` | `max_angle_deg` (default 45), `max_area` | area ≤ max_area |
| `min_wall` | `min` | thickness ≥ min |

- Any check on a part that didn't build fails with `"part '<n>' did not build"`. A measurement `error` fails with that error.
- Reasons for visible checks state the measured value and the target (e.g. `"volume 5821.4 mm³, expected 6000 ±1%"`).
- `render_report(results: list[CheckResult], problems: list[str], truncated: bool) -> str`, exactly:
  - first line `Check report for your last reply:`;
  - then `- format: <problem>` for each parse problem;
  - then `- PASS <check> (<part>)` or `- FAIL <check> (<part>): <reason>`, **visible checks only**;
  - if truncated, a line `- your reply was cut off at the length limit`;
  - last line `Revise and reply with every required part block again.`
- Hidden results never appear.

- [ ] **Step 1: Failing tests:** a table-driven pass and fail case per check; `test_unbuilt_part_fails_all_its_checks`; `test_report_hides_hidden_checks`; `test_report_exact_text` (a golden string).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `mech: checks and feedback report`.

### Task 6: Round driver and the pass

**Files:** Create `src/bakeoff/mech/rounds.py`, `tests/test_mech_rounds.py` (uses `tests/fake_openai.py` and `FakeSandbox`).

**Interfaces:**
- Consumes: `EndpointClient.chat`, `split_think` behaviour (`ChatResult.content` has no reasoning), `ContextPool` and `worst_case_tokens` from `quality.py`, `ErrorBudget("mech")`, and Tasks 1–5.
- `async def run_task(client, suite: MechSuite, item: MechItem, sandbox: Sandbox, pool: ContextPool | None) -> dict` returns the task record:
  - `{"id", "rounds": [round_rec], "final": {"checks": [CheckResult as dict], "passed": bool}, "rounds_used": int, "error": str|None}`
  - `round_rec = {"round", "prompt_sha256", "output", "finish_reason", "latency_s", "prompt_tokens", "completion_tokens", "reasoning_tokens", "problems", "facts", "visible": [results], "error"}`
- Loop:
  1. Send `first_messages`.
  2. Parse the reply. If there are no problems, run the sandbox with the job built from the reply's parts plus `item.given`.
  3. Evaluate. If every visible check passes and there are no problems, stop early. Otherwise, if rounds remain, append `{"role": "assistant", "content": content}` and `{"role": "user", "content": render_report(...)}` and continue.
  4. `final` is the last round's full evaluation. The task passes iff that round has no problems and every check passes.
- Failures:
  - A chat error ends the task with `error` set and `passed: false`.
  - A request whose worst case exceeds `pool.capacity` is not sent: the round's `error = "context"` and the task ends failed.
  - A sandbox failure (`ok == False`) fails `builds` for every part with the sandbox error, which feeds back normally.
- `async def run_mech(client, suites: list[MechSuite], sandbox: Sandbox, context_tokens: int | None) -> MechResult` with `@dataclass class MechResult: records: dict[str, list[dict]]; suites: list[dict]`.
  - Up to `MECH_CONCURRENCY` tasks run at once, and sandbox calls are gated by `SANDBOX_CONCURRENCY`.
  - Records are kept in item order. AbortRun and cancel attach `partial`, as `run_quality` does.
- Suite summary: `{"key", "version", "split", "hash", "n", "items_total", "passed", "pass_rate", "ci95" (wilson), "errors", "rounds_used": {"1": n, "2": n, "3": n}, "checks": {name: [passed, total]}}`.

- [ ] **Step 1: Failing tests:**
  - `test_passes_first_round_stops_early`;
  - `test_fix_on_second_round` (round 1 fails visible `volume`; the report is sent; round 2 passes);
  - `test_hidden_failure_not_reported_but_fails_task`;
  - `test_history_excludes_reasoning` (a fake reply with `<think>…</think>`; the assistant message in history has none);
  - `test_context_overflow_round_error`;
  - `test_format_problem_skips_sandbox`;
  - `test_sandbox_timeout_feeds_back`;
  - `test_summary_counts`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `mech: round driver`.

### Task 7: Runner, config, bundle and publish integration

**Files:** Modify `src/bakeoff/config.py`, `runner.py`, `publish.py`, and `bundle.py` if its verify enumerates folders. Tests in `tests/test_runner.py`, `tests/test_publish.py`, `tests/test_verdicts.py`, `tests/test_config.py`.

**Interfaces:**
- `@dataclass(frozen=True) class MechConfig: suites: list[str] = []; sandbox: str = "bwrap"` as section `[mech]` of `RunConfig`.
- `PASSES = ("quality", "mech", "speed", "fit")`. The `mech` pass runs only when `cfg.mech.suites` is non-empty. It runs after quality and before speed, so no sandbox work overlaps a speed measurement.
- Bundle:
  - files `mechanical/<track>.<name>.jsonl`;
  - meta `"mechanical": {"info_only": true, "sandbox": backend, "versions": facts versions from the first sandbox run, "suites": [summaries]}`, present only when the pass ran;
  - `meta["suites"]` (the suite list with hashes) includes mechanical suites.
- `publish` refuses a bundle whose `mechanical.sandbox != "bwrap"`, with the message `mechanical results from an unsandboxed run can't be published`.

- [ ] **Step 1: Failing tests:**
  - `test_mech_pass_writes_section_and_files` (fake endpoint plus `FakeSandbox` injected through a runner hook `sandbox_factory`);
  - `test_mech_not_in_quality_tracks`;
  - `test_verdicts_ignore_mechanical` (a week with two runs differing only in mechanical results gives identical picks and verdicts);
  - `test_publish_refuses_unsandboxed_mech`;
  - `test_verify_accepts_mechanical_folder`;
  - `test_config_mech_defaults`;
  - `test_mech_pass_skipped_without_suites`.
- [ ] **Step 2–4:** FAIL → implement → PASS; then the full suite: `uv run pytest -q`.
- [ ] **Step 5: Commit** `runner: the mech pass, information only`.

### Task 8: CLI, CI and the authoring guide

**Files:** Modify `src/bakeoff/cli.py`, `.github/workflows/ci.yml`, `README.md` (Known limits); create `docs/mechanical-authoring.md`, `tests/test_mech_cli.py`.

**Interfaces:**
- `bakeoff mech check-suite <dir> [--sandbox subprocess|bwrap]`:
  - loads the suite and runs every item's `good` reference through parse → sandbox → evaluate;
  - exits 0 only if every `good` passes every check and every `bad` fails at least one hidden check;
  - prints one line per reference.
- `bakeoff mech regrade <bundle-dir> [--execute]`:
  - re-evaluates every task from saved `facts` (pure);
  - with `--execute`, re-runs the sandbox on saved outputs and compares facts (floats within 1e-6 relative);
  - exits 0 when identical, and prints differences otherwise.
- CI: a second job `mech`, which runs `uv sync --locked --extra mech`, then `BAKEOFF_MECH_UNSANDBOXED=1 uv run pytest -q -m kernel`, then `BAKEOFF_MECH_UNSANDBOXED=1 uv run bakeoff mech check-suite tests/data/mech/edit-demo --sandbox subprocess`, then the same for `fit-demo`.
- `docs/mechanical-authoring.md` gives the owner a walkthrough:
  - the item fields;
  - each check's params, from the Task 5 table;
  - writing `good` and `bad` references;
  - running `check-suite`;
  - the rule that tasks never carry client or professional specifics.

- [ ] **Step 1: Failing tests:** `test_check_suite_demo_ok` (kernel), `test_check_suite_flags_weak_bad` (a `bad` that passes everything → exit 1), `test_regrade_identical` (from a saved bundle fixture, no kernel).
- [ ] **Step 2–4:** FAIL → implement → PASS; run the full local CI (`bash ~/.local/state/drydock/2026-11-16/scripts/ci.sh <worktree>`, plus the `mech` job's commands).
- [ ] **Step 5: Commit** `mech: check-suite and regrade commands, CI job, authoring guide`.

---

## After this plan

- **The owner writes the first real suites:** `suites/mechanical/edit` (≈8 tasks) and `suites/mechanical/fit` (≈8), following the authoring guide. I build the references, and `check-suite` must pass in CI.
- **Part 2 (K5) and Part 3** (anchor, site display, athena enablement, host setup) get their own plans.
