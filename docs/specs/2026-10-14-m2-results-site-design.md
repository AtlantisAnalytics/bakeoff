# M2 results and site design

Status: design approved 2026-10-14, awaiting spec review. Scope: milestone M2 in [PLAN.md](../../PLAN.md).
It builds on the [M1 harness design](2026-10-13-m1-harness-design.md).

## Goal

`bakeoff publish week.toml` turns one weekend's run bundles into a published week. It:
- verifies every bundle;
- computes a replace / hold / watch verdict for each candidate, per tier and track;
- writes `results/<week>/`;
- regenerates the static site and JSON API in `site/`.

Merging that change to `main` deploys it through Workers Builds. Anyone can rerun `publish` on the
committed bundles and get byte-identical output.

Until launch (M5), the deployed site stays the holding page, with a newsletter sign-up form.

## Decisions

1. **Verdicts come from rules, with an override.** Public code computes every verdict from published
   rules. The owner can override one, but the site then shows both the computed verdict and the
   owner's, with the written reason.
2. **Generate and commit.** `publish` writes the HTML, JSON and feed into `site/`, and they are
   committed. There is no build step on Cloudflare, so the PR diff is exactly what goes live. A
   check fails when `site/` differs from what `results/` regenerates.
3. **Static HTML with no JavaScript.** Pages are generated in Python, and every value is
   HTML-escaped. A Content-Security-Policy forbids all scripts.
4. **Raw data lives in one place.** Run bundles live only under `results/`. Pages link to them in
   the GitHub repo instead of copying them into `site/`.
5. **Launch gate.** While `launched = false`, the deployed site is the holding page plus sign-up,
   and published weeks exist only in `results/` and in a local preview.
6. **No personal data on our side.** The newsletter form posts directly to a hosted newsletter
   service, with double opt-in and with open and click tracking off. The owner creates that account.
   The site also offers an Atom feed.
7. **Deterministic output.** Nothing reads the clock or the environment while generating, and every
   timestamp comes from `week.toml`. The bootstrap uses a documented PRNG, so another language can
   reproduce it.

## CLI

| Command | What it does |
|---|---|
| `bakeoff publish <week.toml>` | Verifies the bundles, copies them into `results/<week>/`, computes the verdicts, writes `week.json`, updates `picks.json`, then regenerates `site/` |
| `bakeoff site` | Regenerates `site/` from `results/` and `site.toml` |
| `bakeoff site --check` | Regenerates into a temporary directory and exits 1 with a diff summary if `site/` differs |
| `bakeoff site --preview <dir>` | Builds the full site as though `launched = true` into a git-ignored directory, for viewing locally |

`publish` refuses to overwrite an existing week. To correct a week, add a `[correction]` table to
its `week.toml` (see Corrections).

## Tiers: `tiers.toml`

The file sits at the repo root, is public, and is the single source for tier definitions.

```toml
[[tier]]
id = "12gb"
name = "12 GB"
envelope = "1× RTX 3060 12 GB"

[[tier]]
id = "24gb"
name = "24 GB"
envelope = "1× RTX 3090 Ti 24 GB"

[[tier]]
id = "36gb-unified"
name = "36 GB unified"
envelope = "Apple M4 Max 36 GB unified"
```

- A bundle belongs to a tier only when its `config.model.envelope` equals the tier's `envelope`
  exactly.
- Tier ids are slugs. Tier ids and envelope strings are never reused for a different meaning.

## Publish input: `week.toml`

M3's bot writes this file. People can also write it by hand. Paths are relative to the file.

```toml
week = "2026-10-19"                     # a Monday (ISO date); publish rejects any other weekday
published_at = "2026-10-19T07:00:00Z"   # the only timestamp, used for the feed and the pages

[[run]]
bundle = "runs/20261018T...-qwen3-14b-q4-k-m-..."
tier = "24gb"
license = "Apache-2.0"                  # an SPDX id, or "other" plus license_url
source_url = "https://huggingface.co/..."

[[judge]]                               # optional; displayed, never used in a verdict
bundle = "runs/judge-..."

[[override]]
tier = "24gb"
track = "general"
run = "20261018T...-qwen3-14b-q4-k-m-..."   # run_id
verdict = "hold"
reason = "Known tokenizer bug in this quant; wait for the re-upload."

[correction]                            # only when re-publishing an existing week
note = "Re-graded structured/json after a grader fix; no verdict changed."
```

Validation fails closed, naming the field, when:
- a field is unknown or a required field is missing;
- a tier doesn't exist;
- an override names a run that isn't in this week;
- a `reason` is empty.

## Verdict rules (rules v1)

The rules are versioned. `week.json` records `rules.version` and every threshold, so a week can
always be re-derived under the rules it was published with.

| Parameter | v1 value |
|---|---|
| `min_context` | 16384 |
| `min_decode_tps` | 10.0 (median `decode_tps` in `short@1`) |
| `speed_ratio` | 0.8 |
| `context_cap` | 32768 |
| `resamples` | 10000 |
| `ci` | 0.95 |

Each run in a tier is assessed separately for each track it has scores in.

1. **Eligibility.**
   - The run must:
     - have `status == "complete"`;
     - pass `bakeoff verify`;
     - reach at least `min_context` in `fit.max_context_passed`;
     - reach at least `min_decode_tps` in short@1 decode.
   - An ineligible run is listed under "tested, doesn't fit", with machine-readable reasons, and gets
     no verdict.
2. **The pick's control run.** For each tier × track with a current pick, `week.toml` must include
   a run of that same model configuration, matched on `weights`, `quant` and `runtime`. If it's
   missing, every candidate in that tier × track is `not_comparable`, which caps its verdict at
   `watch`, and the pick is shown as "not re-measured this week".
3. **Comparability.**
   - The candidate and the pick must have the same set of suites in that track, with equal suite
     hashes.
   - Items are paired by `id`.
   - Any mismatch makes the pair `not_comparable` (capped at `watch`), with the reason recorded.
4. **Track score and difference.**
   - Following M1, a track score is the mean of its suites' accuracies.
   - The observed difference is `score(candidate) − score(pick)`.
   - For the CI, each of the `resamples` iterations does this for every suite in the track:
     1. draw `n` item indices with replacement from that suite's `n` paired items;
     2. compute `accuracy_c − accuracy_p` over the drawn items.

     The iteration's statistic is the mean of those per-suite differences.
   - The CI is taken from the sorted statistics at indices `floor(0.025·B)` and `ceil(0.975·B) − 1`.
5. **PRNG.**
   - The generator is splitmix64.
   - The seed is the first 8 bytes, big-endian, of
     `sha256("bakeoff-verdict-v1|" + week + "|" + tier + "|" + track + "|" + candidate.bundle_sha256 + "|" + pick.bundle_sha256)`.
   - An index is `next() mod n`. The modulo bias is negligible for n < 2³², and it is documented.
   - A golden test vector pins the first ten outputs.
6. **Verdict.**
   - **replace** requires all of:
     - the CI's lower bound is above 0;
     - the candidate's short@1 decode is at least `speed_ratio` × the pick's;
     - the candidate's `max_context_passed` is at least `min(pick's, context_cap)`.
   - **watch** applies when either:
     - the observed difference is above 0 but the CI's lower bound is at most 0;
     - the CI's lower bound is above 0, but the speed or context check fails;
     - or the pair is `not_comparable`.
   - **hold:** everything else.
   - The rules record each reason, for example `ci_includes_zero`, `slower_than_ratio`,
     `less_context` or `suite_hash_mismatch`.
7. **No current pick (first week, or a new track).** The eligible run with the highest track score
   becomes the `initial_pick`. Ties go to higher short@1 decode, then to the lower `run_id` in
   byte order.
8. **More than one replace.** Among the final `replace` verdicts in a tier × track, the highest track
   score becomes the new pick, with the same tie-breaks. The others keep `replace` and get
   `outranked_by: <run_id>`.
9. **Override.** It sets `final` and keeps `computed`. An overridden `replace` counts for rule 8; an
   override away from `replace` removes the run from it.
10. **Picks update.**
    - `picks.json` is rewritten from the new picks.
    - Each entry records `tier`, `track`, the model fields, `since_week` and the `bundle_sha256` of
      the run that won.
    - The pick changes only through a merged week.

The `general/open-ended` judge suite has no accuracy, so it's outside track scores, as in M1.
When judge bundles exist, the page shows their win, tie and loss counts as information only.

## Lenient JSON score

- The `json_schema` grader adds `extractable: bool` to its grade. It is true when the strict check
  fails but a JSON value taken from the output passes the same schema and value checks.
- Where the JSON may come from:
  - exactly one fenced code block (with or without a `json` language tag);
  - exactly one `<json>…</json>` element;
  - the span from the first `{` or `[` to its matching closing bracket.
- A strictly passing item is also extractable.
- Suite summaries for `json_schema` suites gain `extractable_accuracy` and `extractable_ci95`.
- `bundle_format` becomes `"m2"`. `publish` accepts `m1` and `m2` bundles, and shows `extractable`
  as "not measured" for `m1`.
- Verdicts never use `extractable`. Pages show it beside the strict score, labelled "strict" and
  "extractable".

## `week.json` (schema `bakeoff.week.v1`)

```
schema          "bakeoff.week.v1"
week, revision, published_at
corrections     [{revision, published_at, note}]
rules           {version, min_context, min_decode_tps, speed_ratio, context_cap, resamples, ci}
harness         [{version, git_commit}]          distinct values across the runs
tiers[]:
  id, name, envelope
  pick_status   {track: "re-measured" | "not_re-measured" | "none"}
  runs[]:
    run_id, bundle_sha256, bundle_path        path relative to the repo root
    model         {label, weights, quant, runtime, license, license_url?, source_url}
    role          "pick" | "candidate"
    eligible, ineligible_reasons[]
    quality       {tracks: {track: {score, suites: [{key, version, hash, n, passed, accuracy,
                   ci95, extractable_accuracy?, extractable_ci95?}]}}}
    speed         {cell: {ttft_s, prefill_tps, decode_tps, aggregate_tps}}  medians, plus min/max/n
    fit           {max_context_passed, server_context}
    memory        {peak_bytes?, size_vram?}      only measured or reported values
    verdicts      {track: {computed, final, override_reason?, diff, diff_ci95, reasons[],
                   outranked_by?, new_pick}}
  judge[]:        {a, b, wins, ties, losses, errors, judge_model, bundle_sha256}
```

- Floats are written in full. Rounding is for display only.
- `schema/week.v1.json` is a JSON Schema document describing this format. It stays inside the
  subset M1's checker supports, so tests validate every generated week against it.

## Results layout

```
results/
  picks.json
  2026-10-19/
    week.toml        the input, exactly as given
    week.json
    runs/<run_id>/   full bundles, byte-identical copies
    judge/<id>/
```

- `publish` copies each bundle and re-verifies the copy (hashes plus scrub).
- It refuses to publish when there is any finding, when any `status` isn't `complete` or
  `incomplete`, or when a `run_id` appears twice.
- Incomplete runs are published, but they are ineligible.

## API (`site/api/v1/`)

| Path | Contents |
|---|---|
| `latest.json` | The newest week, in full |
| `weeks/index.json` | `[{week, revision, published_at, url}]`, newest first |
| `weeks/<week>.json` | One week |
| `picks.json` | The current picks |
| `schema/week.v1.json` | The JSON Schema |

**Stability.**
- v1 URLs never change.
- Within v1, fields are only ever added.
- A breaking change goes to `/api/v2/`, and v1 stays frozen with its last data.

**Corrections.**
- A correction raises `revision` and appends to `corrections`, and the week is regenerated.
- Git keeps the earlier revision.
- `picks.json` is recomputed. If a correction would change a pick that a later week already
  depends on, `publish` refuses, and the fix has to be a new week.

## Site (`site/`)

The pages are generated from string templates in `src/bakeoff/site/`, with `html.escape` on every
interpolated value.

| Path | Page |
|---|---|
| `/` | Holding page with sign-up while `launched = false`; the latest week once it is true |
| `/weeks/<week>/` | The week page |
| `/weeks/` | The archive |
| `/method/` | Tiers, rules v1 with their thresholds, the suites, and how to reproduce a week. Hand-written as `site_src/method.html` and copied as-is |
| `/privacy/` | What the sign-up collects, who stores it, how to unsubscribe, and that the site has no analytics or cookies. Hand-written as `site_src/privacy.html` |
| `/feed.xml` | An Atom feed with one entry per week, `id = tag:bakeoff.si,2026:week/<week>`, and `updated` set from the latest revision |
| `/_headers` | Security headers (see below) |

**Week page.**
- There is one section per tier.
- **Pick row:**
  - model label, quant, runtime and license;
  - each track score with its CI;
  - short@1 decode tok/s, TTFT and usable context;
  - a link to the raw bundle.
- **Candidate rows** add the verdict for each track, the difference with its CI, and the reasons.
  An override shows as "Computed: hold, owner: replace", followed by the reason.
- "Tested, doesn't fit" lists each run with its reasons.
- Structured JSON shows both "strict" and "extractable".
- The page ends with a footer linking to the method page, the JSON for this week, the feed and the
  sign-up form.

**Display formats.** Scores are percentages to one decimal place, CIs are shown as ±, tok/s has one
decimal place, TTFT is in ms, and context is shown in K. Every rounding happens in one place, so the
formats stay consistent.

**Static assets.** Each asset has a single home:
- **Hand-maintained, never touched by the generator:** the logo, the icons, `og.jpg`, the method and
  privacy page sources, and the stylesheet. The stylesheet is extracted from the current holding
  page into `site/style.css`.
- **Generator-owned:** `index.html`, `weeks/`, `api/`, `feed.xml` and `_headers`.

`--check` compares only the generator-owned paths. It fails on stray files under them.

**Raw-bundle links.** A link goes to `{repo_url}/tree/main/{bundle_path}`. `repo_url` comes from
`site.toml`.

### `site.toml`

```toml
launched = false
base_url = "https://bakeoff.si"
repo_url = "https://github.com/AtlantisAnalytics/bakeoff"

[newsletter]
action = ""            # the form's POST URL from the newsletter service; empty hides the form
email_field = "email"  # the input's name, as the service expects it
```

### Headers (`site/_headers`)

These are generated, and they apply to every path:

- `Content-Security-Policy:`
  - `default-src 'none'; img-src 'self'; style-src 'self'; base-uri 'none'; frame-ancestors 'none';`
  - `form-action <origin of newsletter.action>`. When there is no form, this is `'none'`.
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`

Because of this policy, styles can't be inline, so the stylesheet moves into `style.css`.

`/api/v1/*` also gets `Access-Control-Allow-Origin: *`, so other sites can fetch the data.

### Newsletter

- The form is a plain `<form method="post" action="...">`: one email input, a submit button, and a
  link to `/privacy/` next to it.
- The provider handles double opt-in, storage, unsubscribe and sending.
- **The owner's setup, outside the repo:**
  - create the account;
  - switch off open and click tracking;
  - add the provider's SPF and DKIM records in Cloudflare DNS;
  - put the form's POST URL into `site.toml`.
- The default suggestion is Buttondown. Any provider that accepts a plain form POST works.

## Deploys

- Workers Builds deploys `main` with no build command. Non-production branch builds are off.
- `wrangler.jsonc` doesn't change.
- **First Git deploy.** This is the holding page with `style.css`, `_headers` and, if the account
  exists by then, the form. After it, check that:
  1. bakeoff.si and www still resolve and redirect as before;
  2. workers.dev is still off;
  3. the response headers include the CSP;
  4. the files under `results/` still return 404.

## CI

`.github/workflows/ci.yml` runs on pull requests and on pushes to `main`. The steps are:
1. `uv sync --locked`;
2. `ruff check` and `ruff format --check`;
3. `pytest`;
4. `bakeoff site --check`;
5. `bakeoff verify` on every bundle under `results/`;
6. `scripts/scan_repo.py`.

Third-party actions are pinned by commit SHA, and the workflow has the read-only `contents: read`
permission. The CI check is advisory in M2. The owner can make it required in `protect-main` once it
has run green.

## M1 follow-ups included

1. `make_redactor` ignores API keys shorter than 8 characters, and known placeholders (`ollama`,
   `EMPTY`, `none`, `sk-no-key-required`), so they don't over-redact.
2. `cli._public_suite_text` catches every exception while loading a suite, and treats the suite as
   private.
3. When a username is a common word that's also used in suite text, the username scan is skipped
   only for model `output` fields. Every other field is still scanned.
4. The spec's sample `run.toml` uses bare suite names, matching the code.

## Testing

- **Verdict rules.** Table-driven tests cover every branch of rules 1–9, using synthetic per-item
  grades: replace, watch for each reason, hold, `not_comparable`, an initial pick, an outranked run,
  and both directions of override.
- **Bootstrap.**
  - A golden vector for splitmix64 and the seed derivation.
  - A fixed-input CI golden value.
  - Identical paired data gives a difference of 0 and a CI of [0, 0].
- **Publish.**
  - Publishing a week from fixture bundles twice gives byte-identical output.
  - These are refused:
    - a corrupted bundle;
    - a scrub finding;
    - a duplicate `run_id`;
    - an unknown tier;
    - a non-Monday week;
    - an existing week without `[correction]`.
  - A correction path test covers both the allowed case and the refused case, where a later week
    depends on the pick.
- **Schema.** Every generated week validates against `schema/week.v1.json`.
- **Site.**
  - A model label containing `<script>` or `"` comes out escaped.
  - With `launched = false`, there are no `weeks/` or `api/` paths, and `/` is the holding page.
  - The form is absent when `action` is empty, and `form-action` is then `'none'`.
  - `--check` detects a hand edit and a stray file.
  - The feed parses as XML and has stable entry ids.
- **Lenient JSON.** Fenced, tagged and bare-brace outputs; two code blocks (not extractable); a
  strict pass also counting as extractable; and an `m1` bundle shown as "not measured".
- **Follow-ups.** One regression test each.

## Out of scope for M2

- The M3 bot, scouting, the weekend schedule, and the draft-PR flow that writes `week.toml`.
- Choosing the judge model and its budget.
- Real weeks, which begin with the M4 dry runs, and flipping `launched` (M5).
- Dark mode, client-side filtering, charts, and pages for each model.
- Revisiting the rule thresholds, which waits until after the first real pass.
