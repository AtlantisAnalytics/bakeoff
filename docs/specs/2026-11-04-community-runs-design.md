# Community runs: contributors' hardware in the Monday report

Status: direction agreed 2026-11-04. This is the umbrella spec: it records the decisions and splits
the work into six sub-projects, and each sub-project gets its own design spec before it is built. It
extends [the M1 harness design](2026-10-13-m1-harness-design.md) and
[the M2 results and site design](2026-10-14-m2-results-site-design.md). It replaces the PLAN.md item
"community submissions from the open harness, shown as distributions".

## Goal

Anyone with a GPU or a unified-memory machine can run the week's tests with the open harness and send
the results in a pull request. Accepted results appear in that Monday's report next to ours.

In order of priority:

1. **Coverage:** results for cards and machines we don't own.
2. **Corroboration:** independent re-runs on the envelopes we do own, so readers can see how much a
   result moves from host to host.
3. **A contributor base** that grows because taking part is precise, credited and easy to check.

Precision comes first. A community result must be as exactly specified, as reproducible and as
verifiable as one of ours, and the report must never present it as more than it is.

## Decisions

| # | Decision | Chosen over |
|---|---|---|
| 1 | **A staged trust ladder.** Every envelope starts with distributions only. An envelope can *graduate* to a community tier with verdicts once its history meets public rules. | Never influencing picks; community tiers from day one |
| 2 | **Submissions are GitHub pull requests,** credited to the contributor's GitHub handle (a pseudonym is fine). One account is one contributor. | An upload service with its own accounts |
| 3 | **A weekly window.** A public week plan fixes what to run before any results exist, and a deadline decides which pull requests make that Monday. | Rolling submissions at any time |
| 4 | **Two profiles.** *quick* is speed and fit (about 5–25 min per model); *full* adds the public quality suites (about 20–40 min). Only *full* counts toward graduation. | Everything always; speed and fit only |
| 5 | **Everything lives in this repo.** Plans, submissions, checks and the report stay together, so one audit trail covers them and `rederive` still works from the repository alone. | A separate community repo; Athena ingesting and closing pull requests |
| 6 | **Git is the ledger; raw files live in a content-addressed store.** Git keeps each bundle's small files and the hash of every file. Raw files (quality outputs, telemetry) go to a public Cloudflare R2 bucket, named by sha256, with a yearly snapshot to a second archive. It applies to all new bundles, ours included. Published weeks keep their current layout. | Everything in git; Git LFS; yearly data repos |

Two rules follow from these:

- **Community runs on an envelope that is already one of our tiers corroborate. They don't drive its
  verdicts.** Our tiers keep comparing runs on our own host, pick re-measured, as rules v1 says.
  Graduation is for envelopes we don't run.
- **The judge is never part of a community run.** It is paid API usage and information-only. Private
  suites never leave our hosts.

## The trust ladder

| Stage | What the report shows | How an envelope gets there |
|---|---|---|
| 0. Community envelope | For each model: the median and spread of decode, four-stream throughput, TTFT and usable context, the number of contributors, and a link to every bundle. Quality from *full* runs, information only. | One accepted submission |
| 1. Community tier | As our tiers: a pick, verdicts and "doesn't fit", marked as community-measured, with the contributors credited | It meets the graduation rules, and the owner confirms it in a `tiers.toml` pull request |

The graduation rules are defined in sub-project 6. They build on three kinds of evidence:

- **Anchors.** Each week plan names anchor models, which are our current picks, and every *full*
  submission must include them. Anchors make submissions comparable with each other and with our runs,
  across weeks.
- **Independent agreement.** Several contributors, judged by account, whose results for the same
  model agree within published tolerances.
- **Spot checks.** The owner can reproduce any submission on rented hardware. Graduation can require
  that one was done.

## Sub-projects

| # | Sub-project | Scope | Depends on |
|---|---|---|---|
| 1 | **Ledger and store** | The new bundle layout. `verify`, `publish` and `rederive` read both layouts. The R2 bucket, the post-merge upload workflow and the yearly archive. | — |
| 2 | **Week plan** | `plans/<week>.toml`: models, exact weight revisions, engine pins, suite hashes, run settings, anchors and the deadline. Athena opens it as a bot pull request each Friday. | — |
| 3 | **Contributor runner** | `bakeoff run --plan … --profile quick\|full`:<br>• installs the pinned engine;<br>• downloads the weights at the exact revision;<br>• serves each model that fits, runs it and stops it;<br>• derives the envelope from the measured hardware.<br>`bakeoff submit` publishes the raw files and opens the pull request. | 1, 2 |
| 4 | **Submission review** | `bakeoff check-submission` in pull-request CI checks hashes, scrub, pins, suite hashes, the deadline, the anchors and the envelope. It posts a checklist, so a review is a short read. | 1, 2 |
| 5 | **Monday community section** | A `community` block in `week.json` (additive, so `/api/v1` stays) and a community section on the week page | 4 |
| 6 | **Graduation** | The rules, their computation in `verdicts`, the method text and the confirmation flow | 5, and some weeks of data |

They are built in that order. Sub-project 1 is useful on its own: it stops the repository growing
with our own weekly bundles too.

## Cross-cutting rules

- **Scrub.** A contributor's bundle passes the same `bakeoff verify` as ours. Environment capture is
  already an allowlist that never records hostnames, users, paths or addresses. A GitHub handle is the
  only identity anywhere in a submission.
- **License.** Every submission is CC BY 4.0 like the rest of `results/`. The contributor agrees to
  that in the pull request with a sign-off line, and the check refuses a pull request without one.
- **Exact pins.** Weights at a full revision hash, from allowlisted publishers, never with
  `trust_remote_code`. Engines at the plan's pinned versions. Suites at the plan's hashes. A run that
  differs in any of these is refused, not flagged.
- **A measured envelope.** The runner derives the envelope string from the GPUs it measured; the
  contributor never types it. Machines that can't be measured, or are measured differently, are left
  out until the runner supports them.
- **Credit.** Every community result names its contributor's handle and links to its pull request.

## Risks

| Risk | Mitigation |
|---|---|
| Fabricated or edited numbers. A hash proves a file is unchanged after the run, not that the run was honest. | Robust statistics (median, spread) with outliers flagged. Anchors cross-checked against our runs. Spot-check reproductions. Graduation needs several independent contributors. The owner can exclude an account, and the report says so. |
| The raw-file store is unavailable | Every raw file is addressed by hash and in the yearly archive, so any copy verifies. The Monday report only needs what git holds. |
| Review load | Checks do the verifying; the review is reading the checklist. If volume grows, trusted reviewers can approve. |
| Contributor setup is hard | The runner installs pinned engines and weights itself. A container image per engine can follow. |
| Engine or driver differences across hosts | Each bundle records them. Distributions are per envelope and per engine version, and the report shows the spread instead of hiding it. |

## Out of scope

- An upload service, contributor accounts outside GitHub, and payment of any kind.
- Engines and formats the harness doesn't run yet (llama.cpp and GGUF are their own milestone).
- Moving published weeks to the new layout.
