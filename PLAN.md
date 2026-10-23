# bakeoff.si plan

Status as of 2026-10-15: **M0 to M3 done; M4 starts the weekend of 2026-10-17** (the private runner passed its live acceptance on both tiers). The repo is seeded, and the holding page has been live at
https://bakeoff.si since 2026-10-12. It was uploaded by hand; deploying from Git is set up in M2. The repo
stays **private until the first public Monday** (M5).

## Goal

A weekly, open benchmark of new open-weight models on consumer hardware. Results are published every
Monday morning. For each memory tier, each candidate gets a verdict (**replace / hold / watch**)
against the current pick, backed by quality, speed and fit numbers. Anyone can reproduce the numbers
with the open harness.

What makes it different from existing boards:
- a weekly cadence on the newest models and quantizations;
- quality and speed together;
- a recommendation for each card, not only a score;
- open method and raw data.

## Boundary: what is public and what stays private

| This repo (public at M5) | Private orchestration (the owner's Athena system) |
|---|---|
| Harness: quality evals, speed and fit measurement, environment capture | Scouting new releases (Hugging Face, quantizers, runtime release notes) |
| Results schema and published results | The weekend schedule: taking a production lane out, serving candidates on a lab port, restoring it |
| The site and its JSON API | The Monday standup in Telegram, with a **[Publish]** button |

**Publishing flow.** The private side produces a scrubbed results bundle. The owner presses
[Publish], and a bot opens a **draft PR** here. The owner reviews and merges, and the merge deploys
the site. The bot never pushes to `main` and has no merge path; the rulesets enforce this.

## Scrub rules (for everything committed here)

- **No internal hostnames, LAN IPs, ports, usernames or local paths.** Describe hardware by its
  envelope instead, e.g. "1× RTX 3090 Ti 24 GB", "3× RTX 3060 12 GB", "Apple M4 Max 36 GB unified".
- **No secrets, tokens or API keys**, not even in examples. Use `.env.example` with placeholder values.
- **No raw run logs in `results/`, only the scrubbed bundle.** Raw output stays in the git-ignored `runs/`.
- **Before M5, re-scan the full history**, not just the tip, and squash it if anything slipped in.
- **Image metadata.** `brand/bakeoff-logo.png` carries a C2PA content-credentials block, which says
  it was generated with an AI image tool. It holds no personal data. The site's re-encoded assets
  don't have it. Before M5, the owner decides whether to keep it.

## Method

The method is specified in [docs/specs/2026-10-13-m1-harness-design.md](docs/specs/2026-10-13-m1-harness-design.md).
In short:

- Quality is measured on every host that runs the model, and verdicts compare runs on the same host,
  item by item. Speed and fit are measured per hardware envelope, with the median and spread over
  repeated runs.
- Every run records its environment by allowlist, with provenance, in a hashed bundle. Weights come
  from allowlisted publishers only, and `trust_remote_code` is never used.
- Engines: a model runs on the engine its format needs (vLLM; ExLlamaV3 via TabbyAPI for EXL3).
  Engines compete for the same pick, every result names its engine and version, every engine keeps
  an unquantized KV cache (the model's own FP16 or BF16), and only pinned upstream engine releases are used.

## Hosting (decided)

- **Domain and DNS.** bakeoff.si is registered at Namecheap. DNS runs on Cloudflare, with DNSSEC
  on (DS published 2026-10-12).
- **Site.** Cloudflare Worker `bakeoff`, static assets only (`wrangler.jsonc`, assets in `site/`).
  - Custom domains: `bakeoff.si` (canonical) and `www`, with www redirecting 301 to root.
  - Always Use HTTPS is on.
  - workers.dev and preview URLs are off.
- **Branding.** Light-only, because the logo ships on an opaque white field. A dark mode needs a
  transparent-background logo first. Regenerate the assets with `brand/make_assets.py`.

## Milestones

| | Scope |
|---|---|
| **M0** ✅ | Repo seed: README, plan, licenses, the holding page in `site/`, the logo source and asset script |
| **M1** ✅ | Harness: a Python + uv package with a `bakeoff run` CLI against an OpenAI-compatible endpoint. Quality and speed/fit suites, environment capture, hashed run bundle, tests |
| **M2** ✅ | Results schema v1. The site renders results from JSON, with no framework. `/api/v1/latest.json` plus per-week files. Git deploys via Workers Builds |
| **M3** ✅ | Private side: the weekend schedule, the Monday standup and [Publish] opening a draft PR here (scouting follows later). Live acceptance passed 2026-10-15 on both tiers; its draft PRs were closed unmerged |
| **M4** | Two or three private dry-run weekends; review the report quality and the scrub. The first is the weekend of 2026-10-17 |
| **M5** | First public Monday: scrub the history, make the repo public, announce |

Later:
- a Verified tier on rented consumer cards (more envelopes, including AMD and Intel);
- community submissions from the open harness, shown as distributions;
- a predictive model across hardware × model × quantization that estimates unmeasured cells and picks
  what to measure next;
- edge devices and energy per token;
- **the llama.cpp engine** (a pinned upstream `llama-server` release), which adds GGUF, the format most
  people run locally;
- **the wrapper tax:** the cost of the convenience layers (Ollama, LocalAI, LM Studio) against the engine
  each one wraps, measured with the same model, quantization and card. Each result records the wrapper's
  version and the engine and version underneath it, and keeps an unquantized KV cache. Environment capture works
  inside containers.

## Repo admin to finish

- [x] Give the GitHub App installation access to this repo.
- [x] Rulesets, imported from the Athena repo's exports:
  - `protect-main`: PR + 1 approval, no force push or deletion;
  - `bot-branches-only`: bots may create and update `athena/**` branches only.
- [x] Connect the `bakeoff` Worker to this repo in Workers Builds. **Check at the first Git deploy that
  the custom domains survive and workers.dev stays off.**
- [x] After merging M2, check the first Git deploy (the site stays the holding page until M5). All six checks done 2026-10-14:
  1. bakeoff.si and www resolve, and www redirects with a 301 to the root.
  2. workers.dev is still off.
  3. `curl -sI https://bakeoff.si/` shows the CSP header.
  4. `https://bakeoff.si/_headers` and `https://bakeoff.si/results/` return 404.
  5. The holding page looks unchanged.
  6. `https://bakeoff.si/_headers` returns 404 (a `_headers` file is applied by Cloudflare, never served).

## Open questions

- ~~Results to site~~ Decided in M2: `bakeoff publish` generates `site/` (pages and `site/api/v1/`)
  from `results/`, and the generated files are committed. There is no build step on Cloudflare, and CI
  fails when `site/` is stale.
- ~~Results schema versioning~~ Decided in M2: `/api/v1` is additive-only. v1 URLs never change, fields
  are only added, and a breaking change goes to `/api/v2/`.
- Which judge model, and its budget per weekend. It must be paid API usage, never a consumer
  subscription.
