# bakeoff.si

New open-weight models are tested every week on ordinary consumer GPUs, and the results are published
Monday morning at **[bakeoff.si](https://bakeoff.si)**. For each memory tier the question is the same:
which model should run on *this* card right now?

**Status: pre-launch.** The site is a holding page, and the harness and results format are being built.
See [PLAN.md](PLAN.md).

## What gets measured

- **Quality.** Task evals and a blind pairwise LLM judge against the current pick. Quality depends on
  the model, the quantization and the runtime, not the GPU, so it is measured once.
- **Speed and fit, per card.** Tokens/s at 1 and N concurrent requests, time to first token, peak
  VRAM, and the largest usable context.
- **The practical facts.** License, weight format and runtime support.

Each memory tier gets a weekly verdict for each candidate: **replace / hold / watch**.

## Layout

| Path | What |
|---|---|
| `src/bakeoff/` | The harness: a Python package with the `bakeoff` command |
| `suites/` | The public eval suites (items, manifests, attribution) |
| `docs/` | The design spec and implementation plans |
| `scripts/` | Maintenance scripts: rebuilding the benchmark slices, locking suites, the repo scrub scan |
| `examples/` | A sample `run.toml` with placeholder values only |
| `site/` | The static site served at bakeoff.si, a Cloudflare Worker with static assets (`wrangler.jsonc`) |
| `tiers.toml` | The memory tiers: id, display name and the hardware envelope that maps to each |
| `site.toml` | Site settings: the launch gate, base URL, repo URL and the sign-up form |
| `site_src/` | Hand-written pages (`method.html`, `privacy.html`) and the stylesheet, copied into `site/` |
| `results/` | Published results, licensed CC BY 4.0 |
| `brand/` | Source logo and `make_assets.py`, which regenerates the site's logo, icons and preview image |

## Publishing a week

A week is published from a `week.toml` that lists the verified run bundles to include:

```toml
week = "2026-10-19"
published_at = "2026-10-19T07:00:00Z"   # every timestamp in the output comes from here

[[run]]
bundle = "../final/<run-id>"            # relative to this file
tier = "36gb-unified"                   # an id from tiers.toml
license = "Apache-2.0"
source_url = "https://huggingface.co/<org>/<model>"
```

Optional `[[judge]]`, `[[override]]` and `[correction]` tables add judge bundles, a reviewed verdict
override with its reason, and a correction note.

```sh
uv run bakeoff publish week.toml      # verify, copy bundles into results/<week>/, compute verdicts,
                                      # write week.json and picks.json, then regenerate site/
uv run bakeoff site --check           # exit 1 if site/ differs from what results/ regenerates
uv run bakeoff site --preview runs/preview   # build the full site as though launched, into a git-ignored dir
python3 -m http.server -d runs/preview 8000
```

`publish` refuses to overwrite an existing week; to fix one, add a `[correction]` table. `site/` is
generated and committed, so the PR diff is exactly what deploys, and CI fails when it is stale. `--root DIR`
points `publish` and `site` at another checkout-shaped directory.

**Launch gate.** While `launched = false` in `site.toml`, the deployed site stays the holding page with
the sign-up form, and published weeks exist only in `results/` and in `--preview`. Flipping it to
`true` is the M5 launch step.

**API.** Once launched, `site/api/v1/` serves `latest.json`, `weeks/index.json`, `weeks/<week>.json`,
`picks.json` and `schema/week.v1.json`. v1 URLs never change and fields are only ever added; a breaking
change goes to `/api/v2/`.

## Running the harness

The harness talks to any OpenAI-compatible endpoint (vLLM, llama.cpp server, Ollama or an MLX server)
that is already serving one model. It never downloads or loads weights. Run it on the same host as the
model server, so it can sample memory and power.

```sh
uv sync
cp examples/run.toml runs/local.toml   # runs/ is git-ignored; edit the endpoint, model and [model] facts
uv run bakeoff run runs/local.toml
uv run bakeoff verify runs/<run-id>
```

A minimal `run.toml`:

```toml
[endpoint]
base_url = "http://localhost:8000/v1"   # scrub-ok; placeholder, never written to the bundle
model = "served-model-name"

[model]
label = "Example-14B-Instruct AWQ"
weights = "hf:example-org/Example-14B-Instruct-AWQ@<revision>"
quant = "awq-int4"
runtime = "vllm"                  # vllm | llamacpp | ollama | mlx | other
envelope = "1x RTX 3090 Ti 24 GB" # hardware by its envelope only

[quality]
suites = ["general", "format", "structured", "grounding", "longctx"]

[speed]
concurrency = [1, 4]
repeats = 5
warmup = 1

[fit]
steps = [4096, 8192, 16384, 32768, 65536, 131072]
```

- `bakeoff run CONFIG [--only quality|speed|fit] [--suite PATH] [--set key=value] [--out DIR]` writes a
  hashed bundle to `runs/<id>/` and exits 0 only when the run is complete.
- `bakeoff verify DIR [--denylist FILE] [--no-username]` checks the bundle's hashes and scans it for
  scrub violations (hostnames with ports, private IPs, home paths, the current username and any
  denylist terms). It exits non-zero on any finding. The username check is skipped for generic
  accounts (`root`, `ubuntu`, `runner` and similar), for names of three characters or fewer, and for
  names that are ordinary words in the public suites (a notice says so); `--no-username` turns it off.
  A bundle with a `private/` folder holds private-suite outputs, and verify says so: do not share it.
- `bakeoff env [--config run.toml]` prints exactly what the environment capture records.
- `bakeoff judge A B --max-usd N` runs the blind pairwise judge over two bundles' `general/open-ended`
  outputs. It needs a paid API key in `BAKEOFF_JUDGE_API_KEY`, plus the judge's base URL, model and
  prices (flags or `--judge-config`). The spend is capped by `--max-usd`.

**What is recorded.** The environment capture is an allowlist: the GPU or chip model, memory size,
driver and runtime versions, the operator's `[model]` facts and the harness version, each with its
provenance (`measured`, `reported`, `operator` or `not_captured`). It never records the endpoint URL,
the API key or its variable name, the hostname, the username, IPs, ports, file paths, environment
variable values, or GPU serials and UUIDs.

### Known limits

- The pairwise judge is blind, but a model can name itself in its answers.
- Run bundles and judge output should go through `bakeoff verify` before they are shared. A bundle
  with a `private/` folder holds private-suite outputs and must not be shared.
- Output caps are 4096 tokens for the reasoning-heavy suites (`gsm8k`, `ifeval`, `structured/json`,
  `grounding/context-only`) and 2048 for the rest and for the fit pass. A model that needs more
  thinking than that is scored as truncated (a fail) in quality; in the fit pass a truncated depth
  without the key is inconclusive and ends the run as incomplete, never a fail.
- A judge call that fails is recorded as an `error` item, not a tie, and the judge stops as
  incomplete when more than 20% of its calls fail.
- The `structured/json` suite checks values on unambiguous fields only. Output must be one JSON value,
  optionally in a code fence. A model that wraps its JSON in other tags fails the item.
- The IFEval slice implements 16 instruction types, in strict mode only.
- Concurrency numbers depend on the server's own parallelism setting. A server that queues requests
  serially shows the queueing in time to first token.

### Validation

Measured on 2026-10-14 with IBM Granite 3.3 8B Instruct (`granite3.3:8b`, Q4_K_M) served by Ollama on
an Apple M4 Max 36 GB unified machine. These are measurements of a public model.

**Speed against the runtime's own counters.** Ollama's native `/api/chat` reports `eval_count` and
`eval_duration`. For the short speed prompt with 256 output tokens, three repeats each, with the
harness's request and the runtime's own request interleaved so that machine-speed drift hits both:

| Check | Harness (client side) | Ollama counters | Ratio |
|---|---|---|---|
| Decode tok/s, run 1 | 57.1 | 57.5 | 0.99 |
| Decode tok/s, run 2 | 56.6 | 56.8 | 1.00 |
| Decode tok/s, run 3 | 55.4 | 55.5 | 1.00 |

All three are within the 15% tolerance. The absolute rate on this machine drifts by 10 to 20% from one
run to the next (and was lower during the 40-minute full run, below), which is why the check
interleaves the two measurements. The harness's time to first token was 0.04 s, against 0.02 s for
the runtime's own prompt evaluation. It is the larger of the two, as expected: it also includes the
HTTP round trip and the first token. Prefill rates are informational only, because Ollama's prompt
cache makes repeated prompts nearly free.

**Quality against published scores.** Source: the Evaluation Results table of the model card at
https://huggingface.co/ibm-granite/granite-3.3-8b-instruct (evaluated by IBM with OLMES), which lists
GSM8K 80.89 and IFEval 74.82 for Granite-3.3-8B-Instruct. The card does not state the shot count or the
IFEval variant.

| Suite | Harness (n, 95% CI) | Published |
|---|---|---|
| `general/gsm8k` | 87% (87/100, 79 to 92) | 80.89 |
| `general/ifeval` | 81% (81/100, 72 to 87) | 74.82 |

The harness scores are within about 6 points of the published ones, and the published figures sit
inside the intervals. The setups differ: a 100-item slice, our prompt, our 16-type strict IFEval
subset, and Q4_K_M quantization instead of the original weights. Expect the same ballpark, not a
match. An earlier run with a 1024-token output cap truncated 4 IFEval items and scored 79%; raising
the cap to 4096 tokens removed every truncation (0 of 100) and moved the score to 81%.

Other suites for the same run: `format/exact-reply` 85%, `grounding/context-only` 97%,
`longctx/needle` 6/6, `structured/json` 40% (all 18 failures were the
model wrapping its JSON in `<json>` tags, which the strict grader rejects, not a grader bug). The
largest context that passed the fit pass was 32768 tokens, with the server reporting a 64000-token
window (probed again just before the fit pass, once the model was loaded), and the larger steps were
skipped as above that limit. Decode was about 48 to 50 tok/s at both concurrency 1 and 4, because this
Ollama setup queues concurrent requests (time to first token at concurrency 4 was about 8 s).

## License

Code: [Apache-2.0](LICENSE). Results data in `results/`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
