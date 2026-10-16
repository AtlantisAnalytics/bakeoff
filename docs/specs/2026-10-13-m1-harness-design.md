# M1 harness design

Status: design approved 2026-10-13, awaiting spec review. Scope: milestone M1 in [PLAN.md](../../PLAN.md).

## Goal

`bakeoff run` points at any OpenAI-compatible endpoint (vLLM, llama.cpp server, Ollama or an MLX
server) that is already serving one model. It measures quality, speed and fit, captures the
environment, and writes a hashed run bundle to `runs/`. The private weekend schedule (M3) drives it,
and anyone can rerun it on their own hardware with the same command.

The harness never downloads or loads weights. Choosing, fetching and serving them is the operator's
job, or the private orchestration's.

## Decisions

1. **The harness runs on the same host as the model server.** It samples memory and power locally,
   so a remote endpoint still works, but its memory and power fields are `not_captured`.
2. **Built for outside contributors from the start.** Environment capture is an allowlist that never
   records the hostname, username, paths or IPs. The harness installs natively with uv (Apple
   unified memory isn't visible from a container), and a Linux/NVIDIA container image can come later.
3. **Every captured field records its provenance**, so a reader can tell what was observed from what
   was claimed.
4. **Quality is scored per track** (`general` now, domain tracks later). A specialist model is never
   judged on one blended general score.
5. **Suites have `public` or `private` splits.** Private suites live outside the repo. The bundle
   records their hash and scores but never their item text or outputs. Private items get rotated
   into the public split after a while (the cadence is decided later, outside M1).
6. **Small runtime footprint.** `httpx` is the only runtime dependency, on Python 3.11 or later. The
   rest is the standard library, including `tomllib` and a small JSON-schema subset checker.
   Development uses pytest and ruff, managed with uv.

## CLI

| Command | What it does |
|---|---|
| `bakeoff run <run.toml>` | Runs the quality, speed and fit passes against one endpoint and model, and writes `runs/<id>/` |
| `bakeoff judge <bundle-a> <bundle-b>` | Blind pairwise judge over two bundles' `general/open-ended` outputs. It writes `runs/judge-<a>-vs-<b>/` |
| `bakeoff env` | Prints the allowlisted environment capture on its own, so a contributor can see exactly what gets recorded |
| `bakeoff verify <dir>` | Checks the hashes and scans for scrub violations. It exits non-zero on any problem |

Flags on `run` can override any `run.toml` key. Useful ones: `--only quality|speed|fit` and
`--suite <path>` (repeatable).

### run.toml

```toml
[endpoint]
base_url = "http://localhost:8000/v1"   # scrub-ok; used for the run, never written to the bundle
model = "served-model-name"
api_key_env = "BAKEOFF_ENDPOINT_API_KEY" # optional; the env var's name, never the key itself
timeout_s = 600

[model]                                  # operator facts the harness can't discover
label = "Example-14B-Instruct AWQ"
weights = "hf:example-org/Example-14B-Instruct-AWQ@<revision>"
weights_file = ""                        # e.g. the GGUF filename
quant = "awq-int4"
runtime = "vllm"                         # vllm | llamacpp | ollama | mlx | other
envelope = "1x RTX 3090 Ti 24 GB"

[quality]
suites = ["general", "format", "structured", "grounding", "longctx"]
private_suites = []                      # absolute paths outside the repo

[speed]
concurrency = [1, 4]
repeats = 5
warmup = 1

[fit]
steps = [4096, 8192, 16384, 32768, 65536, 131072]
max_context = 0                          # optional operator cap; 0 = use what the server reports
```

## Package layout

All code lives in `src/bakeoff/`, with one job per file:

| File | Job |
|---|---|
| `config.py` | Loads and validates `run.toml` and the CLI overrides |
| `client.py` | Async OpenAI-compatible client: streaming, TTFT timing, usage capture, retries, and runtime version probes |
| `env.py` | Allowlisted environment capture |
| `sampler.py` | Background sampler for memory and GPU power |
| `suites.py` | Loads suites (manifest, items, generators) and hashes them |
| `graders.py` | The grader registry |
| `quality.py` | Quality pass |
| `speed.py` | Speed pass |
| `fit.py` | Fit pass |
| `stats.py` | Median and min–max, Wilson intervals |
| `bundle.py` | Writes bundles, hashes them, and runs verify and the scrub scan |
| `judge.py` | Pairwise judge |
| `cli.py` | argparse wiring, plus a `bakeoff` entry point in `pyproject.toml` |

Other paths:
- `suites/`: the public suites (see below).
- `scripts/make_slices.py`: rebuilds the public benchmark slices.
- `tests/`: the tests.
- `examples/run.toml`: a sample config with placeholder values only.

## Suites

A suite is a directory containing:

- `manifest.toml`:
  - `track`, `name`, `version`, `split` (`public` or `private`);
  - `license` and `source` (upstream, revision, attribution);
  - `grader` and its options;
  - `max_tokens`;
  - optional `system`.
- `items.jsonl`: one item per line, with `id`, `messages` (or `prompt`) and `expected` (shaped per grader).
  A generated suite (the needle) names its generator and a seed in the manifest instead of shipping
  items.

The suite hash is SHA-256 over the canonical manifest plus the items, or plus the generator name and
parameters for a generated suite. It changes whenever any item changes. Editing a suite also requires
bumping `version`; otherwise results from the two versions would look comparable when they aren't.

### Suites in M1

| Suite | Grader | Size | Source |
|---|---|---|---|
| `general/gsm8k` | `numeric` (the final number in the answer) | 100 | GSM8K test split, MIT |
| `general/ifeval` | `ifeval` (a subset of the instruction checkers) | ~100 | IFEval, Apache-2.0. Only items whose instructions all fall in the implemented checker subset are kept |
| `general/open-ended` | `judge` (graded later by `bakeoff judge`) | 20 | Ours: practical assistant tasks |
| `format/exact-reply` | `exact` | 20 | Ours: "Reply with exactly: …" and short fixed formats |
| `structured/json` | `json_schema` | 30 | Ours: extraction and classification into a given schema |
| `grounding/context-only` | `grounding` (required facts present, forbidden facts absent, refuses when the answer isn't in context) | 30 | Ours: fictional systems-status documents |
| `longctx/needle` | `needle` | generated | Ours: a deterministic filler text plus a random key |

`scripts/make_slices.py` downloads each upstream file at a pinned revision, checks its SHA-256 and
samples it with a fixed seed. It writes `items.jsonl` and an `ATTRIBUTION.md` next to it. The licenses
named above are confirmed against the pinned revision when the slices are built, and the script
records the license text's location in the manifest.

### Graders

Each grader returns `{pass: bool, score: 0..1, reason: str}`.
- `exact`: whitespace is trimmed, and nothing else may differ.
- `numeric`: the final number in the answer, with a configurable tolerance.
- `regex`: the pattern must match.
- `json_schema`: the output must parse as one JSON value (an optional code fence is allowed) and
  validate against a subset of JSON Schema: `type`, `properties`, `required`, `enum`, `items` and
  `additionalProperties`.
- `ifeval`: the subset of instruction checkers.
- `grounding`: required facts present, forbidden facts absent, and a refusal when the answer isn't
  in the context.
- `needle`: the key must appear in the answer.
- `judge`: grading is deferred to `bakeoff judge`.

An `executable` grader (for example, a Verilog testbench) is out of scope for M1, but the registry
must allow adding one without changing other graders.

## Quality pass

- **Requests:** every request uses temperature 0, a fixed seed and the suite's `max_tokens`. These
  limits are set generously, because reasoning models can use up a tight budget before they answer.
- **Grading:** only the final message `content` is graded. Reasoning tokens (`reasoning_content`, or
  the usage breakdown) are counted separately when the server reports them.
- **Truncation:** `finish_reason == "length"` is a fail with the reason `truncated`.
- **Recorded per item:**
  - `id`, a hash of the prompt, the output;
  - reasoning tokens, `finish_reason`;
  - the grade, the latency;
  - `error`, if any.
- **Scores:** each suite gets an accuracy score with a 95% Wilson interval, and a track's score is the
  mean of its suites.
- **Not M1:** paired comparison against the incumbent and the replace/hold/watch verdict. They work
  from the per-item grades.

## Speed pass

- **Prompts:** two fixed prompt shapes, short (~128 prompt tokens) and long (~2K), each asking for
  256 generated tokens. The prompts are written so that output reliably runs long. Where the runtime
  accepts it (vLLM, llama.cpp), `ignore_eos` is sent and recorded.
- **Repeats:** each (shape × concurrency) cell runs `warmup` throwaway rounds, then `repeats` measured
  rounds.
- **Metrics:**
  - TTFT;
  - prefill tok/s (`prompt_tokens / TTFT`);
  - decode tok/s per request (`(completion_tokens - 1) / (t_last - t_first)`);
  - aggregate tok/s at N (total completion tokens / wall time of the round).

  Each is reported as the median with min–max over the measured rounds.
- **Token counts:** taken from `usage` via `stream_options.include_usage`. If the server doesn't report
  usage, the harness counts streamed content chunks, and the metric's provenance is `estimated`.

## Fit pass

For each step in `fit.steps` up to the context the server reports, send a needle prompt sized to the
step, with the needle at 10%, 50% and 90% depth. The server-reported context comes from vLLM
`/v1/models` `max_model_len`, llama.cpp `/props` `n_ctx`, Ollama `/api/ps` `context_length`, or else
`operator` (`fit.max_context`). If none of those is available, every step is tried, and the first
failing step ends the pass.

A step passes only when all three of these hold:
- the server accepts the request;
- `usage.prompt_tokens` is within 5% of the expected count, which catches silent truncation such as
  Ollama's `num_ctx`;
- the needle is retrieved.

Record the TTFT and peak memory for each step. The fit result is the largest step that passes.

For vLLM, the context limit and the KV cache are fixed when the server starts. The harness measures
what the current serving config allows, and trying other serving configs is the orchestrator's job.

## Memory and power sampling

The sampler starts before the quality pass and stops after the fit pass, sampling every 0.5 s. It
keeps a separate peak for each pass.

| Platform | Memory | Power |
|---|---|---|
| NVIDIA (Linux) | `nvidia-smi --query-gpu=index,memory.used,power.draw`, per GPU and summed | `power.draw` per GPU |

On a multi-GPU host, set `CUDA_VISIBLE_DEVICES` to the run's GPUs, as for its server. The sampler and
the environment capture then query only those GPUs (`nvidia-smi -i`), so the bundle neither lists
nor sums cards the run didn't use. Unset or unparseable, they cover every GPU.
| macOS | `vm_stat`: wired + active + compressed pages. Plus Ollama `/api/ps` `size_vram` when the runtime is Ollama | `not_captured` (needs `sudo powermetrics`) |
| Other | `not_captured` | `not_captured` |

The bundle reports the raw peak and what the runtime itself reports: vLLM's KV-cache usage % from
`/metrics`, or Ollama's `size_vram`. vLLM pre-allocates most of the card, so its raw peak reflects
`gpu_memory_utilization` rather than what the model needs. The site has to explain that difference.

Power samples are stored. Energy per token is a later feature.

## Environment capture

Each field is stored as `{"value": ..., "source": "measured" | "reported" | "operator" | "not_captured"}`.
Only these fields exist:

- **Host:**
  - OS family and version;
  - CPU model and core count;
  - total RAM.
- **GPU (each GPU):**
  - name and VRAM;
  - driver version, CUDA version;
  - power limit.

  On a Mac, this is the chip name and unified memory instead.
- **Runtime:**
  - name and version (`reported` from vLLM `/version`, llama.cpp `/props`, Ollama `/api/version`;
    otherwise `operator`);
  - served model name (`reported`).
- **Model (all `operator`):**
  - label, weights reference and file;
  - quantization, envelope.
- **Harness:**
  - version;
  - git commit;
  - whether the working tree was dirty;
  - Python version.

These are never recorded: the hostname, username, IPs, ports, the endpoint URL, file paths, the
environment variables' values, GPU serial numbers and UUIDs, and MAC addresses.

## Bundle

```
runs/<YYYYMMDDTHHMMSSZ>-<model-slug>-<hash8>/
  bundle.json
  quality/<track>.<suite>.jsonl
  private/<track>.<suite>.jsonl     # outputs from private suites; never published
  speed.jsonl
  fit.jsonl
  samples.jsonl
```

`bundle.json` contains:
- `bundle_format` (`"m1"`; v1 of the results schema comes in M2);
- the run ID, start and end times, and `status` (`complete` or `incomplete`, with a reason);
- the config with `endpoint.base_url` and `api_key_env` removed;
- the environment;
- the suite hashes and splits;
- the summary scores, speed, fit and memory;
- `files` (path → SHA-256);
- `bundle_sha256`.

The bundle hash is SHA-256 over the canonical JSON (sorted keys, compact separators, UTF-8) of
`bundle.json`, with `bundle_sha256` itself removed.

The hash detects accidental edits, not deliberate tampering. Community results (later) are shown as
distributions, kept apart from our own numbers.

### verify and scrub

`bakeoff verify` recomputes every file hash and the bundle hash, and it flags `incomplete` bundles. It
then scans every file for:
- private-range IPv4 (RFC 1918, CGNAT, link-local, loopback) and IPv6 (ULA, link-local);
- `host:port` patterns;
- `/Users/`, `/home/` and `C:\Users\` paths;
- the current username;
- every term in an optional local denylist (`~/.config/bakeoff/denylist`, one term per line, never
  committed).

Any finding means a non-zero exit, with the file and line listed.

## Judge

`bakeoff judge A B --judge-model M --max-usd X` takes its other options from flags or a `[judge]`
table in a TOML file:
- the judge base URL (any OpenAI-compatible endpoint);
- the API key, read from `BAKEOFF_JUDGE_API_KEY`;
- input and output prices per million tokens, which are required so the cap can be enforced.

- **Matching suites:** it refuses to run unless both bundles have the same `general/open-ended` suite
  hash.
- **Order swapping:** each item is judged twice, with A/B order swapped. The judge sees only the
  prompt and two anonymous answers, and returns JSON `{winner: "1" | "2" | "tie", reason}`. A win
  counts only when both orders agree; any disagreement is a tie.
- **Budget:** before the first call, it estimates the worst-case cost from the prompt lengths and
  `max_tokens`. If the estimate is over `--max-usd` it refuses, unless `--allow-partial` is given; in
  that case it judges items until the next call could cross the cap.
- **Output:** `judge.json` contains:
  - both bundle hashes, the judge model and the prices;
  - per-item results with reasons;
  - win/tie/loss counts and win rate with a Wilson interval;
  - the actual cost.

  It is hashed like a bundle.
- **Default judge:** chosen in the M4 dry runs (an open question in PLAN.md). Payment is always on API
  keys, never a consumer subscription.

## Errors

- **Preflight:** before anything is measured, `GET /models` must succeed and must list the configured
  model; otherwise the harness exits non-zero.
- **Per request:** a timeout, connection error or 5xx is retried twice with exponential backoff. After
  that the item is recorded as `error`, counted as a fail, and reported separately from wrong answers.
- **Abort:** if more than 20% of a pass's requests end in `error`, the run aborts. The bundle is
  written as `incomplete` with the reason. Ctrl-C does the same.

## Testing

- **Unit tests:**
  - every grader (table-driven);
  - suite loading and hashing (a version bump is required when items change);
  - `stats`;
  - the bundle hash round trip;
  - `verify`, with tampered files and planted scrub violations;
  - `env` parsing from recorded `nvidia-smi`, `sysctl` and `vm_stat` output with fictional values;
  - config validation;
  - the judge's budget logic and order-swap rule.
- **Integration:** a fake OpenAI-compatible server (`httpx.MockTransport` with streamed chunks,
  usage, configurable delays, injected errors and a truncation threshold) drives `bakeoff run` end to
  end. The test asserts that the bundle verifies, that the endpoint URL appears nowhere under the run
  directory, that the incomplete or abort path works, and that the fit pass catches silent truncation.
- **Live smoke test:** opt-in, skipped unless `BAKEOFF_LIVE_ENDPOINT` and `BAKEOFF_LIVE_MODEL` are set.
  Before M1 is called done, it runs once against a local Ollama, with output in `runs/`.
- **Lint and format:** `ruff check` and `ruff format --check` must pass.

## Out of scope for M1

- the replace/hold/watch verdict and paired comparison against the incumbent;
- results schema v1, the publish export and the site (M2);
- containers, executable graders and domain tracks;
- energy per token, scouting and the weekend schedule (M3).
