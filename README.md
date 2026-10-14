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
| `site/` | The static site served at bakeoff.si, a Cloudflare Worker with static assets (`wrangler.jsonc`) |
| `results/` | Published results, licensed CC BY 4.0 |
| `brand/` | Source logo and `make_assets.py`, which regenerates the site's logo, icons and preview image |

## License

Code: [Apache-2.0](LICENSE). Results data in `results/`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
