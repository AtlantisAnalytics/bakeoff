# Ledger and store: small bundles in git, raw files by hash

Status: design approved 2026-11-04, awaiting spec review. Sub-project 1 of
[community runs](2026-11-04-community-runs-design.md). It amends
[the M1 harness design](2026-10-13-m1-harness-design.md) (bundles) and
[the M2 results and site design](2026-10-14-m2-results-site-design.md) (publishing).

## Goal

The repository stops growing with raw run data, and nothing about integrity, reproducibility or the
published numbers gets weaker. Git is the ledger: it holds each bundle's small files and the hash of
every file. Raw files live in a public, content-addressed store, and a yearly copy goes to Zenodo. Any
copy of any file verifies against git.

This applies to our own weekly bundles from a switch-over week on, and later to community
submissions. Published weeks are not migrated.

## Why now

A full run bundle is about 1.3 MB, and 30 KB of it is what the site reads:

| Part of a full run bundle | Size | Read by |
|---|---|---|
| `bundle.json`, `speed.jsonl`, `fit.jsonl` | ~30 KB | `week.json`, the site, the Monday report |
| `quality/*.jsonl` (model outputs, item by item) | ~870 KB | Verdicts at publish, `rederive`, audits |
| `samples.jsonl` (1 s GPU telemetry) | ~430 KB | Peaks already summarised in `bundle.json`; future metrics |

Our own weeks add 5 to 13 MB of bundles. With community submissions, which are expected, it would
be tens to hundreds of MB a week. Moving that layout later would mean migrating published history,
so the switch happens before any contributor bundle arrives.

## Decisions

1. **The bundle doesn't change.** The harness keeps writing complete bundles. `bundle.json` already
   lists every file with its sha256, and `bundle_sha256` covers `bundle.json` itself. The split
   happens only when a bundle is staged into `results/`, so a complete bundle on a disk and its split
   copy in git have the same `bundle_sha256`. No bundle format bump is needed.
2. **What stays in git is set by data, not by code:** `results/layout.toml`, in ranges of weeks.
   Changing the rule later means adding a range. The code never names a date.
3. **The store is Cloudflare R2,** read publicly through `data.bakeoff.si`. Objects are named by
   sha256 and never overwritten. Athena uploads our weekly files directly with a write-only bucket
   token. Fork pull requests never see a credential.
4. **Downloads are untrusted** until their hash matches `bundle.json`, whether they come from R2, the
   local cache or the archive. A mismatch is an error, never a warning.
5. **Withdrawals are possible and public.** A person can delete a stored object, for example a model
   output that reproduced personal data. A row in `results/withdrawn.toml` records the hash, the date
   and the reason. The harness itself can't delete anything.
6. **A yearly snapshot goes to Zenodo:** one record per year, with a DOI, made on or after 1 March
   for the previous year, so withdrawals can happen before anything becomes permanent.
7. **Raw data is never thinned** ([umbrella spec](2026-11-04-community-runs-design.md), "Assessment
   stays adjustable"). Telemetry and outputs are kept in full, so later metrics and re-grading stay
   possible.

## 1. Layout

### `results/layout.toml`

```toml
# Which files of a published bundle are kept in git, by week. Each range applies from its week
# until the next range. Every other file a bundle lists lives in the store, named by its sha256.

[[layout]]
from = "2026-11-02"   # published before the store existed: every file in git
keep = "all"

[[layout]]
from = "2026-11-30"   # added in its own pull request when the switch-over is scheduled
run = ["bundle.json", "speed.jsonl", "fit.jsonl"]
judge = ["bundle.json"]
```

- Ranges are sorted by `from` and must not repeat a week. A week before the first range is an
  error.
- `keep = "all"` means the bundle directory is complete. Otherwise, `run` and `judge` list the
  kept file names for each kind of bundle (a judge bundle is one whose `bundle_format` is
  `judge-m1`). A future kind of bundle gets its own key. A kind the range doesn't name is an error,
  not a default.
- For a split range, a bundle directory must contain **exactly** the kept files the bundle lists: no
  stored file and nothing extra. That is what stops the growth from coming back.
- Community bundles (sub-project 4) are run bundles and follow `run`.

### The resolver

Every reader of a bundle file goes through one function,
`resolve(bundle_dir, rel) -> Path`. `bakeoff.bundle` gains it, and `verify_bundle`,
`runview.load_run` (quality items) and `rederive` use it. Resolution order:

1. the file in the bundle directory, if present;
2. the cache, `$BAKEOFF_CACHE` or else `~/.cache/bakeoff/store/sha256/<aa>/<hash>`;
3. a download from `$BAKEOFF_STORE_URL` or else `https://data.bakeoff.si`, at `/sha256/<aa>/<hash>`
   (`<aa>` is the hash's first two hex digits).

A downloaded file is written to a temporary file, hashed, and renamed into the cache only if its
hash matches the one `bundle.json` lists. Cached files are hashed again when they are read, since
the cache is untrusted too. If the hash is withdrawn (`withdrawn.toml`), the resolver raises
`Withdrawn(date, reason)` without trying the network.

`verify_bundle` changes in one place. A listed file that is missing from the directory is no longer
`missing_file` if the week's layout says it is stored. Instead it is resolved and checked, giving
`hash_mismatch`, `unavailable` (with the URL and HTTP status) or `withdrawn`. An `offline=True`
mode skips resolution and only checks the ledger. CI uses that mode (section 4).

## 2. Store commands and credentials

### `bakeoff store`

| Command | What it does | Needs |
|---|---|---|
| `store fetch <bundle>… [--into DIR] [--archive YEAR]` | Resolves every stored file of each bundle into the cache. `--into` instead writes complete bundle copies, which pass `bakeoff verify` as they are. `--archive` takes files from that year's Zenodo record instead of R2. | Nothing |
| `store push <bundle>…` | Runs a full `verify` on each complete bundle. Then, for every stored file, it skips the upload if the object already exists (`HEAD`), and otherwise checks the file's hash and uploads it. It prints the counts uploaded and skipped. | Write token |
| `store check <bundle>… [--rehash N]` | Sends a `HEAD` for every stored object, and downloads and re-hashes N chosen at random. A withdrawn object is reported as withdrawn, not missing. | Nothing |

### Talking to R2

- **Uploads and existence checks** use R2's S3 API at `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`
  with Signature V4. Signing is about 80 lines of stdlib (`hmac`, `hashlib`, `urllib`), which keeps
  boto3 out of the harness's dependencies. The client implements `PUT` and `HEAD` only; there is no
  delete.
- **Public reads** are plain HTTPS `GET`s from `data.bakeoff.si`.
- **Credentials come from the environment only:** `BAKEOFF_STORE_ENDPOINT`,
  `BAKEOFF_STORE_KEY_ID`, `BAKEOFF_STORE_SECRET` and `BAKEOFF_STORE_BUCKET` (default
  `bakeoff-data`). They are never logged, never read from a file, and never recorded in a bundle
  (environment capture is an allowlist that doesn't include them).
- **Objects are uploaded with** `Content-Type: application/octet-stream` and
  `Cache-Control: public, max-age=31536000, immutable`.
- **Retries:** a transient failure (a network error, HTTP 429 or 5xx) is retried 3 times with
  backoff. Any other status fails at once and reports the object's hash and the status. A re-run of
  `push` is always safe, because existing objects are skipped.

## 3. Publishing

### `bakeoff publish` for a week in a split range

1. Verify every input bundle in full, from the publisher's complete local copies (as today).
2. Run `store check` on every input bundle, and refuse if any stored object is missing. A week can't
   be published with data the store doesn't have.
3. Compute verdicts from the complete local bundles, as today, so the results are the same as an
   unsplit publish.
4. Stage only the kept files into `results/<week>/runs/<id>/` and `results/<week>/judge/<id>/`, then
   check the staged copy again: kept-file hashes, `bundle_sha256`, and the layout rule.
5. Write `week.json` and regenerate `site/`, as today.

Weeks in a `keep = "all"` range publish exactly as they do now. Re-publishing a split week (a
correction) uses the published copy plus the resolver for its stored files.

### Athena's [Publish]

In the private repository: before `bakeoff publish`, Athena runs `bakeoff store push` on the week's
run and judge bundles, with the token held on the publish host. If the push fails, [Publish] stops,
reports the error in Telegram and opens no pull request. Athena's `bakeoff_ref` must point at a
bakeoff commit that includes this sub-project before the first split week.

### `bakeoff rederive <week>`

For a split week, rederive reads the kept files from git and resolves the rest, so it works from a
fresh clone with no credentials. A second run works offline from the cache.

If a file needed for verdicts (a quality item file) is withdrawn, rederive stops and names the file,
the date and the reason. A withdrawn file that verdicts don't need (telemetry, judge reasons)
doesn't stop it.

## 4. CI

### Every pull request (no secrets)

The step "Verify published bundles" becomes two:

1. **`bakeoff verify --ledger results/`** checks every bundle in the repository, fully offline:
   - `bundle_sha256` and the kept-file hashes match;
   - the week's layout range holds;
   - every `withdrawn.toml` row names a hash some bundle lists, with an ISO date and a non-empty
     reason;
   - scrub scanning, as now, on everything in git.

   It does not depend on the network, so CI time stays flat as the history grows.
2. **`bakeoff verify --changed origin/<base>`** fully verifies, through the resolver, every bundle
   directory that the pull request adds or changes. A week pull request whose objects aren't in the
   store fails here, in public.

`bakeoff site --check` and the scrub scan of the repository don't change. Both stay offline.

### Weekly store audit (`store-audit.yml`)

This scheduled workflow on `main` runs every Tuesday and needs no credentials:
- `bakeoff store check` over every bundle in a split range;
- `--rehash 50`.

The audit reads through the public domain, so the edge cache can answer for an object that is gone
from R2. Auditing R2 directly is a later improvement.

When it fails, the workflow goes red, which notifies the owner through GitHub. A Telegram alert from
Athena is a possible later addition and is not part of this sub-project.

**Cost:** each object costs one Class B operation per audit. At about 90,000 objects a year, a
weekly audit is under 400,000 Class B operations a month, well inside R2's 10 million free per
month.

### Not in this sub-project

The post-merge workflow that copies contributor uploads into R2 belongs to sub-project 4. It will
call `store push`.

## 5. Withdrawals and the yearly archive

### `results/withdrawn.toml`

```toml
[[withdrawn]]
sha256 = "<64 hex>"
date = "2027-01-12"
reason = "model output reproduced personal data"
bundles = ["results/2026-12-07/runs/<id>"]
```

**The procedure:**
1. A pull request adds the row, with the reason in plain words.
2. After it merges, the owner deletes the object in the Cloudflare dashboard.
3. Objects are served with `Cache-Control: public, max-age=31536000, immutable` and a cache rule
   keeps them at the edge, so deleting the object does not remove those copies. The owner purges the
   URL `https://data.bakeoff.si/sha256/<aa>/<hash>` (Cloudflare dashboard, bakeoff.si zone, Caching,
   Configuration, Purge Cache, Custom Purge, URL).
4. From then on, `verify`, `fetch` and `rederive` report "withdrawn on <date>: <reason>".

Rows are never removed. Only objects in the store can be withdrawn: anything in git stays in
history. That is one more reason git holds only the small files with numbers in them.

### The yearly archive (`archive.yml`)

This workflow on `main` is started by hand, with the year as its input, on or after 1 March of the
following year.

1. Collect every object listed by that year's split weeks, leaving out withdrawn ones.
2. Write `MANIFEST.sha256` (one line per object: hash, size, and the bundles that list it) and a
   `README.md` (what this is, the license, and how to verify against the repository).
3. Pack the objects into one tar per month (`<year>-<MM>.tar`). A year is at most 12 tar files plus
   the manifest and the README, well under Zenodo's 100-file limit.
4. Upload to Zenodo as one new record, "bakeoff.si raw data, <year>", CC BY 4.0, in a `bakeoff-si`
   Zenodo community. Each year is its own record, linked to the previous year's with a related
   identifier. A year is a separate dataset, not a new version of the same one.
5. Open a pull request adding the record's DOI and the manifest's sha256 to `results/archive.toml`.

**Limits:**
- Zenodo allows 50 GB and 100 files per record. A one-time increase to 200 GB can be requested for
  a single record.
- Our estimate is 1 to 10 GB a year, depending on community volume.
- Zenodo asks people not to split one dataset across records to get around the limit. If a year
  ever approaches 50 GB, we request the increase rather than split.
- Published files can never be changed, and that is the reason for waiting until 1 March.

`bakeoff store fetch --archive <year>` downloads the record's tar files once, unpacks them into the
cache, and resolves from there. Every file is still checked against `bundle.json`.

## 6. Setup, testing and rollout

### Owner setup (walkthrough)

**R2 bucket and public domain**
1. Cloudflare dashboard → R2 → Create bucket: name `bakeoff-data`, location automatic, Standard
   storage.
2. Bucket → Settings → Custom Domains → Connect domain: `data.bakeoff.si`. The bakeoff.si zone is
   already in this account, so Cloudflare adds the DNS record. Wait for the status to show Active.
3. Bucket → Settings → Public Development URL: leave it **disabled**. If it is on, select Disable and
   type `disallow`.
4. bakeoff.si zone → Caching → Cache Rules → Create rule:
   - name: "data immutable";
   - when: hostname equals `data.bakeoff.si`;
   - then: eligible for cache, edge TTL "use cache-control header if present".

   Without the rule, objects without a file extension aren't cached.

**Write token**
1. R2 → Manage API tokens → Create API token:
   - permission: Object Read & Write;
   - scope: only bucket `bakeoff-data`;
   - no expiry, or 1 year with a renewal reminder.
2. Copy the Access Key ID and Secret Access Key. The secret is shown only once. Note the account
   endpoint, `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.
3. Put it in the GitHub repository, Settings → Environments → New environment `store`:
   - deployment branches: `main` only;
   - secrets: `BAKEOFF_STORE_ENDPOINT`, `BAKEOFF_STORE_KEY_ID`, `BAKEOFF_STORE_SECRET`.
4. Put it on the publish host as an Athena secret, in the same place as its other publish
   credentials.

**Zenodo:** not needed before March 2027. Create the account, the `bakeoff-si` community and a
personal access token with the `deposit:write` and `deposit:actions` scopes, then add the token as
`ZENODO_TOKEN` in the `store` environment.

### Testing

- **Resolver:** a local file, a cache hit, a download, a hash mismatch from the network and from the
  cache, a withdrawn hash, and an unreachable store.
- **`layout.toml`:** parsing, range selection, unknown kinds, unsorted or repeated ranges, and the
  exact-files rule. A stored file present in git is refused.
- **`withdrawn.toml`:** validation.
- **Ledger mode:** `verify --ledger` catches each kind of finding without network access.
- **SigV4:** known-answer tests against AWS's published Signature V4 examples.
- **A fake store fixture:** a local HTTP server that implements `PUT`, `HEAD` and `GET` by key, with
  switches for 404, 500 and corrupt bytes. `push`, `check`, `fetch`, a split `publish` and `rederive`
  then run end to end without R2.
- **Regression:** `bakeoff rederive 2026-11-02` still prints `identical`, under the `keep = "all"`
  range.

### Rollout

1. Merge the harness changes with only the `keep = "all"` range in `layout.toml`. Behavior is
   unchanged.
2. The owner does the setup above, and Athena's push step and `bakeoff_ref` bump land in a draft
   pull request on its repository.
3. **Dry run** on a private branch:
   - publish a test week with a test date, pushing its objects to R2;
   - `rederive` it from a fresh clone with no credentials. It must print `identical`;
   - then delete the branch and the test objects.
4. A pull request adds the split range for the first real split week. The target is **2026-11-30**,
   which leaves the weekends of 11-14 and 11-21 for the dry run.
5. On the first split Monday, check the pull request CI, the live week page, and `rederive` from a
   clean clone.

## Out of scope

- The contributor upload path and the post-merge mirror workflow (sub-project 4).
- Versioned verdict-rule logic and `bakeoff regrade` (separate tasks, from the umbrella spec's
  "Assessment stays adjustable").
- Moving published weeks to the new layout.

## Facts checked 2026-11-04

- **R2 pricing.** The free tier includes 10 GB-month of storage, 1 million Class A and 10 million
  Class B operations a month, and free egress. `PutObject` is Class A; `HeadObject` and `GetObject`
  are Class B. ([R2 pricing](https://developers.cloudflare.com/r2/pricing/))
- **R2 public buckets.** A custom domain must be a zone in the same account. `r2.dev` can be
  disabled. Custom domains support Cloudflare cache, and caching every file type needs a rule.
  ([R2 public buckets](https://developers.cloudflare.com/r2/buckets/public-buckets/))
- **R2 tokens.** Object Read & Write tokens can be scoped to particular buckets. They give an Access
  Key ID and a Secret Access Key, with the endpoint
  `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.
  ([R2 API tokens](https://developers.cloudflare.com/r2/api/tokens/))
- **Zenodo limits.** 100 files and 50 GB per record, with a one-time increase to 200 GB on request.
  ([Zenodo size limits](https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo))
- **Zenodo records.** Published files and identifiers can't be changed; a new version is a new
  record. ([Zenodo records](https://help.zenodo.org/docs/deposit/about-records/))
