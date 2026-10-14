# bakeoff.si plan

Status as of 2026-10-13: **M0 done**. The repo is seeded, and the holding page has been live at
https://bakeoff.si since 2026-10-12. It was uploaded by hand; deploying from Git comes in M2. The repo
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

## Method (first draft, to refine in M1)

- **Quality vs speed.** Quality depends on model × quantization × runtime, not on the GPU, so it is
  measured once per combination. Speed and fit are measured per hardware envelope.
- **Quality suite:**
  - task evals;
  - exact-format instruction following ("Reply with exactly: …");
  - JSON or structured-output accuracy;
  - grounding: answer only from the provided context and never invent facts;
  - a long-context needle test where the envelope allows it;
  - a blind pairwise LLM judge against the incumbent.
- **Speed and fit:**
  - tokens/s at 1 and N concurrent requests;
  - time to first token;
  - peak memory;
  - the largest context that fits;
  - repeated runs, reporting the median and the spread.
- **Environment capture** for every run: GPU, driver, runtime and version, quantization, power limit
  and harness version. These are recorded in the bundle and hashed.
- **Weights:**
  - from allowlisted publishers and quantizers only;
  - safetensors or GGUF;
  - **never `trust_remote_code`**.
- **Endpoints.** The harness talks to any OpenAI-compatible endpoint: vLLM, llama.cpp server,
  Ollama or an MLX server.

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
| **M1** | Harness: a Python + uv package with a `bakeoff run` CLI against an OpenAI-compatible endpoint. Quality and speed/fit suites, environment capture, hashed run bundle, tests |
| **M2** | Results schema v1. The site renders results from JSON, with no framework. `/api/v1/latest.json` plus per-week files. Git deploys via Workers Builds |
| **M3** | Private side: scouting, the weekend schedule, the Monday standup and [Publish] opening a draft PR here |
| **M4** | Two or three private dry-run weekends; review the report quality and the scrub |
| **M5** | First public Monday: scrub the history, make the repo public, announce |

Later:
- a Verified tier on rented consumer cards (more envelopes, including AMD and Intel);
- community submissions from the open harness, shown as distributions;
- a predictive model across hardware × model × quantization that estimates unmeasured cells and picks
  what to measure next;
- edge devices and energy per token.

## Repo admin to finish

- [ ] Give the GitHub App installation access to this repo.
- [ ] Rulesets, imported from the Athena repo's exports:
  - `protect-main`: PR + 1 approval, no force push or deletion;
  - `bot-branches-only`: bots may create and update `athena/**` branches only.
- [ ] Connect the `bakeoff` Worker to this repo in Workers Builds. **Check at the first Git deploy that
  the custom domains survive and workers.dev stays off.**

## Open questions

- Results to site: should a build step copy `results/` into `site/api/v1/`, or should the site read
  from a separate path? Decide in M2.
- Results schema: versioning and URL stability guarantees.
- Which judge model, and its budget per weekend. It must be paid API usage, never a consumer
  subscription.
