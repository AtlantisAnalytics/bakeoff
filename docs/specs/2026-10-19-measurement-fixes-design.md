# Measurement fixes before M5

Status: design approved 2026-10-19, awaiting spec review. Scope: four findings from the M4 week-1 review
(the 2026-10-19 standup PR). It amends [the M1 harness design](2026-10-13-m1-harness-design.md).

## Goal

Every number a week publishes means what it says, and a reader can tell which harness version produced
it. Nothing about verdicts changes.

## Findings

| # | Finding | Evidence (week 2026-10-19, Qwen2.5 0.5B bf16, vLLM 0.19.0) |
|---|---|---|
| 1 | The same weights, quant and engine score differently per host | GSM8K 29/100 on 1× RTX 3060 12 GB, 37/100 on 1× RTX 3090 Ti 24 GB; IFEval 29 and 25 |
| 2 | Fit's `kv_cache_usage` is always 0.0 | `fit.py` reads vLLM's live gauge after the step's requests finish, when the cache is idle |
| 3 | The memory peak equals the baseline | 11581 MiB and 23004 MiB: vLLM's up-front reservation (`gpu_memory_utilization`), not what the model needs |
| 4 | Bundles record no harness commit | `git_commit: not_captured`. The orchestrator ships the harness with `git archive`, which has no `.git` |

## Decisions

1. **Quality stays per host, and the difference is explained.**
   - **What we measured:** the quality pass is already deterministic per host (temperature 0, seed 0,
     one request at a time). Across the two hosts, only 32% of GSM8K outputs and 12% of IFEval outputs were
     byte-identical, but pass/fail agreed on 90% and 94% of items. Rounding differs between cards and
     compounds over long outputs.
   - **Why we keep it:** verdicts already compare a candidate with the pick re-measured on the same host,
     item by item (rules v1, rule 2). One shared score would make that pairing cross hosts and add exactly
     this noise.
   - **So:** the method is corrected, and the cross-host agreement is published. No score changes.
2. **KV is reported as the cache's capacity plus a derived need per fit step,** not as a polled gauge.
   Polling misses short requests. A 4k-token request on a fast model finishes between two 0.5 s samples.
3. **The raw memory peak stays in the bundle, unchanged,** and the method page explains that it is the
   engine's reservation. week.json doesn't publish it today, and it still won't.
4. **The harness commit comes from git's `export-subst`,** so the orchestrator needs no change. It also
   covers anyone who runs a GitHub source tarball, because GitHub builds those with `git archive`.

## 1. Quality per host

### Method text

In the M1 harness design and on the method page, replace "quality is measured once per model ×
quantization × runtime" with:

> Quality is measured on every host that runs the model. Floating-point rounding differs between cards, so
> the same weights can produce different text on different hosts and occasionally a different pass/fail. A
> verdict therefore compares a candidate with the current pick re-measured on the same host, item by item,
> and never compares scores across hosts. When a model runs on several hosts in a week, the week reports how
> often their pass/fail agreed.

PLAN.md's Method section says the same thing in one line and changes with it.

### `quality_agreement` in week.json

This is a new, optional top-level array, an additive change to `week.v1.json`.

- **When it's present:** only for models with runs on at least two tiers this week under the same
  `model_key` (weights, quant, runtime), eligible or not. Each tier contributes one run, its lowest
  `run_id`.
- **One entry per model,** sorted by model key:

```json
{
  "weights": "hf:Qwen/Qwen2.5-0.5B-Instruct@7ae5576…",
  "quant": "bf16",
  "runtime": "vllm",
  "runs": ["20261017T160328Z-…", "20261017T160336Z-…"],
  "suites": {
    "general/gsm8k": {"n": 100, "same_pass": 90, "same_output": 32},
    "general/ifeval": {"n": 100, "same_pass": 94, "same_output": 12}
  }
}
```

- **What's counted:**
  - `n` is the number of item ids present in every run's records for that suite. A suite missing from
    any run is left out.
  - `same_pass` counts items whose `pass` is equal across all runs.
  - `same_output` counts items whose `output` string is byte-identical across all runs.
- **What's skipped:** suites without per-item pass (the judged `general/open-ended`) and private suites.
  Neither ever appears in this field.
- **More than two runs:** an item counts as "same" only if every run agrees.
- **Where the data comes from:** the published item records. Everything is re-derivable, and `bakeoff
  rederive` checks the field like any other.

### Site

- **The method page:** one paragraph, the text above.
- **The week page:** when `quality_agreement` exists, show one line per model under the tiers, for example
  "Same model on two cards: pass/fail agreed on 90% (GSM8K) and 94% (IFEval) of items." No chart.

## 2 and 3. KV capacity and fit

### What the harness reads

vLLM exposes `vllm:cache_config_info`, a gauge fixed at 1 whose labels are the whole `CacheConfig` (checked
in 0.19.0, `vllm/v1/metrics/loggers.py` and `CacheConfig.metrics_info`). The harness reads it once, after the
fit pass, through the same `/metrics` request `kv_cache_usage` uses today. It records three labels:
`num_gpu_blocks`, `block_size` and `cache_dtype`.

### Bundle changes (`bundle_format` stays `"m1"`; fields only added)

- **`fit.kv`:** a new object, with each field carrying its provenance:
  - `capacity_tokens`: `num_gpu_blocks × block_size` (`reported`);
  - `block_size` (`reported`);
  - `cache_dtype` (`reported`), for example `"auto"`, which means the model dtype;
  - when the runtime isn't vLLM, or the metric is missing or unparseable, all three are `null` with
    `not_captured`.
- **Each fit step that ran** (status pass or fail): gets `kv_fraction`, defined as
  `ceil(step / block_size) / num_gpu_blocks`, rounded to 4 decimals (`derived`). It is the share of the KV pool that one request of that context size needs.
  `null` when capacity isn't captured.
- **`kv_cache_usage` per step is removed.** Its value was always 0.0, and old bundles keep theirs.
  `runview` reads `kv_fraction` when present and ignores `kv_cache_usage`.

### week.json

`run.fit` gains `kv_capacity_tokens`, `kv_cache_dtype` and `kv_fraction_at_max`, the `kv_fraction` of
`max_context_passed`. All three are optional and `null` when not captured.

### Limits, stated on the method page

- **`kv_fraction` assumes every layer keeps full-attention KV for the whole context.** For sliding-window
  or hybrid models it is an upper bound.
- **The capacity is for this serving config only.** Another `gpu_memory_utilization` or `max_model_len`
  gives another pool; trying configs is still the orchestrator's job.
- **`cache_dtype` lets a reader check the FP16-KV rule.** It is recorded and shown, not enforced. See
  Out of scope.

### Memory text on the method page

> The memory figure is what the engine reserved, not what the model needs. vLLM claims a fixed share of the
> card at start-up and fills it with weights and KV cache. For fit, read the KV capacity and the share of it
> one request of each size needs.

## 4. Harness commit

- **New file `src/bakeoff/_commit`:** a single line, `$Format:%H$`.
- **`.gitattributes`:** `src/bakeoff/_commit export-subst`. `git archive` (and therefore GitHub tarballs)
  writes the commit hash into the file.
- **`env.py`:** when `_git` finds no checkout (it returns `None, None`), read `_commit` next to the module.
  - If the file holds 40 lowercase hex characters, record `git_commit` from it (`reported`, because git
    wrote it at archive time) and `git_dirty: false` (`reported`, because an archive has no working tree).
  - Anything else (the literal placeholder, a missing file, a read error) keeps today's `not_captured`.
  - A real checkout still wins and stays `measured`.
- **Packaging:** hatchling's wheel already takes every file under `src/bakeoff`, so the file ships. No
  config change is needed; a test asserts it.
- **Rollout:** nothing changes in the orchestrator. The commit appears as soon as the lab's pinned
  `bakeoff_ref` points at a commit that contains this file.

## Compatibility

- **Additive only:** `bundle_format` stays `"m1"`, the schema stays `week.v1.json`, and every new field is
  optional. Bundles published before this change still verify, re-derive and render.
- **Verdicts and eligibility don't change.** The new fields are for display only.

## Testing

- **`quality_agreement`:**
  - two and three runs;
  - identical and differing outputs;
  - a suite missing from one run;
  - private and judged suites excluded;
  - sorted, deterministic output;
  - `rederive` re-computes it.
- **KV:**
  - parsing `cache_config_info` (a labelled line, a missing metric, garbage labels, a non-vLLM runtime);
  - `kv_fraction` arithmetic (rounding, the ceil at block boundaries);
  - week.json fields present and absent.
- **Commit:**
  - a checkout (measured);
  - an archive with a substituted hash (reported, dirty false);
  - the placeholder left in;
  - a missing file;
  - `git archive` of the repo really substitutes the hash (a test that runs git when it is available).
- **End to end:** a fake vLLM server with `/metrics` serving `cache_config_info`, then `bakeoff run` and
  `publish` against it.
- **CI:** everything above runs on both 3.12 and 3.14 locally. CI stays pinned to 3.12.

## Out of scope

- Measuring how much memory the model weights take. vLLM logs it but doesn't expose it.
- KV capacity for other engines (llama.cpp, EXL3). They stay `not_captured` until those engines land.
- Skipping the quality pass on a second host. The agreement figure needs both runs.
- Enforcing the FP16-KV rule from `cache_dtype` (for example, an fp8 KV makes a run ineligible). That
  changes eligibility, so it is a rules decision; propose it once a few weeks of `cache_dtype` exist.
