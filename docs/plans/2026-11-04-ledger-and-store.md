# Ledger and Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Published bundles keep only their small files in git, and every other file resolves by
sha256 from a public store (R2 via `data.bakeoff.si`, a local cache, or a yearly Zenodo archive).
Nothing about integrity, verdicts or `rederive` changes.

**Architecture:**
- **Layout and withdrawals:** a data file, `results/layout.toml`, says which files each week keeps in
  git. `results/withdrawn.toml` lists objects taken down.
- **One resolver:** `bakeoff.store.Store` turns a listed file into a verified local path: the bundle
  directory first, then the cache, then a download.
  - `verify_bundle`, `load_run`, `publish` and `rederive` use it.
  - Uploads go through a small stdlib Signature V4 client.
- **CI:** an offline ledger check over all of `results/`, plus full verification of the bundles a
  pull request changes.

**Tech Stack:** Python 3.12, stdlib only (`urllib.request`, `hmac`, `hashlib`, `tarfile`,
`http.server` for the test fake), uv, ruff, pytest, GitHub Actions.

**Spec:** `docs/specs/2026-11-04-ledger-and-store-design.md`. The umbrella is
`docs/specs/2026-11-04-community-runs-design.md`.

## Global Constraints

- No new runtime dependencies. In particular no boto3 and no requests.
- Object key: `sha256/<first two hex>/<64-hex sha256>`. Public base: `https://data.bakeoff.si`.
- Environment variables:
  - `BAKEOFF_STORE_URL` (read base, default `https://data.bakeoff.si`);
  - `BAKEOFF_CACHE` (default `~/.cache/bakeoff/store`);
  - `BAKEOFF_STORE_ENDPOINT`, `BAKEOFF_STORE_KEY_ID`, `BAKEOFF_STORE_SECRET`;
  - `BAKEOFF_STORE_BUCKET` (default `bakeoff-data`).
- Uploads:
  - headers `Content-Type: application/octet-stream` and
    `Cache-Control: public, max-age=31536000, immutable`;
  - SigV4 region `auto`, service `s3`, path-style URL `<endpoint>/<bucket>/<key>`.
- The client implements `PUT`, `HEAD` and `GET` only. No code path deletes an object.
- Retries: network errors, HTTP 429 and 5xx get 3 retries with backoff (1 s, 2 s, 4 s). Every other
  status fails at once with the hash and the status.
- A downloaded or cached file is used only if its sha256 matches `bundle.json`. A mismatch is an
  error, never a warning.
- Credentials are never logged, never read from a file, and never written to a bundle or to output.
- Kept files for split ranges: run `["bundle.json", "speed.jsonl", "fit.jsonl"]`, judge
  `["bundle.json"]`. A bundle is a judge bundle when its `bundle_format` is `judge-m1`, and a run
  bundle otherwise.
- The layout applies to `results/<week>/runs/<id>` and `results/<week>/judge/<id>` only.
  `results/experiments/**` bundles are always complete.
- A root without `results/layout.toml` behaves as `keep = "all"` for every week. Test roots and older
  checkouts depend on that.
- Weekly audit: every Tuesday, `--rehash 50`. Archive: on or after 1 March. One Zenodo record per
  year, one tar per month (`<year>-<MM>.tar`), plus `MANIFEST.sha256` and `README.md`.
- **A deliberate deviation from the spec:** `store push` and `store check` cover *every* listed file,
  not just the stored ones. That keeps them independent of the layout: a later layout change never
  needs a re-push. The extra objects are small.
- Literal loopback addresses in tests need a trailing `# scrub-ok` comment, or `scripts/scan_repo.py`
  flags them.
- Python style: ruff with line length 100. Comments and docstrings sparse, as in `bundle.py`.

## Review Focus

These are the inputs the spec implies but its tests are least likely to exercise:

1. **Shallow CI checkouts.** `actions/checkout` fetches depth 1 by default, so
   `verify --changed origin/main` has no base to diff against. It must fail loudly, never verify
   nothing, and `ci.yml` must fetch enough history. The test goes in Task 8.
2. **The same hash under two names.** For example, two empty `.jsonl` files share
   `e3b0c442…b855`. Resolving, materializing and pushing must handle one object serving two paths,
   and a zero-byte object. The tests go in Tasks 2 and 3.
3. **A cache that can't be written,** or no `HOME` (CI containers). Resolution must fail with the
   path in the message, not a traceback. The test goes in Task 2.
4. **`BAKEOFF_STORE_URL` with a path prefix or trailing slash** (a mirror at
   `https://example.org/mirror/`). Keys must join to exactly one `/`. The test goes in Task 2.
5. **Interrupted downloads.** Killing a download mid-way must leave no file in the cache, and two
   processes fetching the same object must not corrupt it (temp file in the cache directory, then
   `os.replace`). The test goes in Task 2.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/bakeoff/ledger.py` (new) | `layout.toml` and `withdrawn.toml` models; locating a bundle's week and kind. Imports nothing from bakeoff. |
| `src/bakeoff/ledgercheck.py` (new) | `check_ledger`, `changed_bundles`. Kept separate because it imports `bundle`, which imports `store`, which imports `ledger`. |
| `src/bakeoff/store.py` (new) | `Store`: object keys, cache, public `GET`/`HEAD`, `resolve`, `materialize`; store errors |
| `src/bakeoff/s3.py` (new) | Signature V4 signing; `S3Client.put` and `S3Client.head` |
| `src/bakeoff/archive.py` (new) | Yearly manifest and monthly tars; reading an archive into the cache |
| `src/bakeoff/bundle.py` | `verify_bundle` learns `kept`, `offline` and `store` |
| `src/bakeoff/runview.py` | `load_run` reads item files through the store |
| `src/bakeoff/publish.py` | Layout-aware staging; store check before staging |
| `src/bakeoff/rederive.py` | Materializes split bundles; copies `layout.toml` and `withdrawn.toml` into the temporary root |
| `src/bakeoff/cli.py` | `store fetch\|push\|check`, `verify --ledger`, `verify --changed`, `archive build` |
| `results/layout.toml`, `results/withdrawn.toml` (new) | Initial data |
| `.github/workflows/ci.yml`, `store-audit.yml` (new), `archive.yml` (new) | CI |
| `scripts/zenodo_upload.py` (new) | Creates and publishes the yearly Zenodo record |
| `site_src/method.html`, `site/` | One Reproduce line |
| `tests/fakestore.py` (new) | A fake S3 and public store over `http.server` |
| `tests/test_ledger.py`, `test_store.py`, `test_s3.py`, `test_archive.py` (new); `test_bundle.py`, `test_runview.py`, `test_publish.py`, `test_rederive.py`, `test_cli.py` | Tests |

---

### Task 1: Layout and withdrawals model

**Files:**
- Create: `src/bakeoff/ledger.py`, `results/layout.toml`, `results/withdrawn.toml`, `tests/test_ledger.py`

**Interfaces:**
- Produces:
  - `class LedgerError(ValueError)`.
  - `@dataclass(frozen=True) class Range: start: str; keep: dict[str, frozenset[str]] | None`.
    `keep` is `None` for `keep = "all"`.
  - `@dataclass(frozen=True) class Layout: ranges: tuple[Range, ...]`, with
    `kept(self, week: str, kind: str) -> frozenset[str] | None`. That returns `None` for complete,
    and raises `LedgerError` for a week before the first range or a kind the range doesn't name.
  - `load_layout(root: Path) -> Layout`. A missing file gives `Layout((Range("0000-01-01", None),))`.
  - `@dataclass(frozen=True) class Withdrawal: sha256: str; date: str; reason: str; bundles: tuple[str, ...]`.
  - `load_withdrawn(root: Path) -> dict[str, Withdrawal]`, keyed by sha256. A missing file gives `{}`.
  - `bundle_kind(meta: dict) -> str`, returning `"judge"` or `"run"`.
  - `week_of(root: Path, bundle_dir: Path) -> str | None`. It returns the week for
    `results/<YYYY-MM-DD>/(runs|judge)/<id>`, and `None` otherwise (experiments, paths outside
    `results/`).

- [ ] **Step 1: Write the failing tests**

```python
def test_ranges_select_by_week(tmp_path):
    write(tmp_path, LAYOUT)  # the spec's two ranges: 2026-11-02 "all", 2026-11-30 run/judge lists
    lay = load_layout(tmp_path)
    assert lay.kept("2026-11-23", "run") is None
    assert lay.kept("2026-11-30", "run") == {"bundle.json", "speed.jsonl", "fit.jsonl"}
    assert lay.kept("2027-01-04", "judge") == {"bundle.json"}

def test_layout_errors(tmp_path):
    # each case raises LedgerError naming the problem:
    # unsorted ranges, a repeated week, a week before the first range ("2026-10-26"),
    # an unknown kind ("community" on a split range), keep = "some", a non-YYYY-MM-DD from

def test_missing_layout_keeps_everything(tmp_path):
    assert load_layout(tmp_path).kept("2099-01-05", "run") is None

def test_withdrawn_rows(tmp_path):
    # a valid row loads; each of these raises LedgerError: sha not 64 lowercase hex,
    # date not ISO YYYY-MM-DD, empty reason, empty bundles, the same sha twice

def test_week_of_and_kind(tmp_path):
    assert week_of(tmp_path, tmp_path / "results/2026-11-30/runs/x") == "2026-11-30"
    assert week_of(tmp_path, tmp_path / "results/2026-11-30/judge/j") == "2026-11-30"
    assert week_of(tmp_path, tmp_path / "results/experiments/2026-11-multi-gpu/runs/x") is None
    assert bundle_kind({"bundle_format": "judge-m1"}) == "judge"
    assert bundle_kind({"bundle_format": "m2"}) == "run"
```

- [ ] **Step 2:** Run `uv run pytest tests/test_ledger.py -q`. Expected: FAIL, because
  `bakeoff.ledger` doesn't exist.
- [ ] **Step 3:** Implement `ledger.py` per the Interfaces block, using `tomllib`. Write
  `results/layout.toml` with only the first range (`from = "2026-11-02"`, `keep = "all"`), keeping
  the spec's comment header. Write `results/withdrawn.toml` with only a comment explaining the
  row format and the procedure (spec section 5).
- [ ] **Step 4:** Run `uv run pytest tests/test_ledger.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit with `ledger: layout.toml ranges and withdrawn.toml`.

### Task 2: Store read side, cache and fake store

**Files:**
- Create: `src/bakeoff/store.py`, `tests/fakestore.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `Withdrawal` (Task 1).
- Produces:
  - `class StoreError(Exception)`, with subclasses `Unavailable(sha, url, status)`,
    `HashMismatch(sha, where)` and `Withdrawn(sha, date, reason)`.
  - `object_key(sha: str) -> str`.
  - `@dataclass class Store: base_url: str; cache: Path; withdrawn: dict[str, Withdrawal]`.
  - `Store.from_env(root: Path) -> Store`. It reads `BAKEOFF_STORE_URL` and `BAKEOFF_CACHE`, and
    `load_withdrawn(root)`.
  - `Store.resolve(self, sha: str) -> Path`. It returns a cached path whose content hashes to `sha`.
  - `Store.exists(self, sha: str) -> bool`, a public `HEAD`, with retries.
  - `Store.cached(self, sha: str) -> Path | None`.
  - `resolve_file(bundle_dir: Path, rel: str, sha: str, store: Store | None) -> Path`. It returns
    `bundle_dir / rel` if that is a regular file, and otherwise `store.resolve(sha)`. With no store
    and no local file, it raises `Unavailable`.
- Test fixture: `tests/fakestore.py` provides `FakeStore`, a context manager. It runs a
  `ThreadingHTTPServer` on the loopback interface, port 0.
  - Attributes: `url`, `objects: dict[str, bytes]` (key to bytes), `requests: list[tuple[str, str, dict]]`
    (method, path, headers), and `fail: dict[str, int]` (key to forced status).
  - It serves `GET` and `HEAD` at `/<key>` and `/<bucket>/<key>`, and accepts `PUT` at
    `/<bucket>/<key>`.
  - It also provides a pytest fixture `fake_store`, which sets `BAKEOFF_STORE_URL` and `BAKEOFF_CACHE`
    (to `tmp_path/cache`) through `monkeypatch`.

- [ ] **Step 1: Write the failing tests** (in `test_store.py`)

```python
def test_object_key():
    assert object_key("ab" + "0" * 62) == "sha256/ab/ab" + "0" * 62

def test_resolve_downloads_once_then_uses_cache(fake_store):  # 1 GET, then 0
def test_resolve_refuses_wrong_bytes(fake_store):             # HashMismatch; nothing left in cache
def test_cached_file_rehashed_on_read(fake_store):            # corrupt the cached file → HashMismatch
def test_withdrawn_raises_without_network(fake_store):        # Withdrawn(date, reason); 0 requests
def test_404_is_unavailable_with_url(fake_store):             # message contains the URL and 404
def test_5xx_retried_then_ok(fake_store, monkeypatch):        # fail twice with 503, then serve; sleep patched
def test_base_url_with_path_and_slash(tmp_path):              # "https://x.org/mirror/" → ".../mirror/sha256/.."
def test_empty_object(fake_store):                            # sha of b"" resolves to a 0-byte file
def test_interrupted_download_leaves_no_cache_file(fake_store, monkeypatch):
    # make the body read raise mid-stream; assert no file under cache/sha256 and no temp file left
def test_unwritable_cache_names_path(tmp_path, monkeypatch):  # StoreError message contains the cache path
def test_resolve_file_prefers_local(tmp_path, fake_store):    # local file used; 0 requests
```

- [ ] **Step 2:** Run `uv run pytest tests/test_store.py -q`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `store.py`.
  - Downloads stream to a `NamedTemporaryFile(dir=<cache>/sha256/<aa>, delete=False)`, hashing as
    they go. Only on a hash match does `os.replace` move the file into its final name; otherwise
    the temp file is unlinked.
  - The base URL is joined with `rstrip("/") + "/" + object_key(sha)`.
  - Retries and timeouts are as in Global Constraints. Use a 60 s socket timeout.
- [ ] **Step 4:** Run `uv run pytest tests/test_store.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit with `store: hash-checked resolver with a local cache`.

### Task 3: Verification, run loading and materializing through the store

**Files:**
- Modify: `src/bakeoff/bundle.py` (`verify_bundle`), `src/bakeoff/runview.py` (`load_run`,
  `_read_items`), `src/bakeoff/store.py` (add `materialize`)
- Test: `tests/test_bundle.py`, `tests/test_runview.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `Store`, `resolve_file`, `StoreError` and its subclasses (Task 2).
- Produces:
  - `verify_bundle(run_dir, username, denylist, username_skip_output=False, *, kept: frozenset[str] | None = None, offline: bool = False, store: Store | None = None) -> list[Finding]`.
  - New finding kinds: `unavailable`, `withdrawn`, `kept_mismatch`.
  - `load_run(path: Path, store: Store | None = None) -> RunView`.
  - `materialize(bundle_dir: Path, dst: Path, store: Store) -> Path`. It writes a complete bundle
    copy at `dst` and returns `dst`.

**Behavior:**
- **`kept` governs membership only.** With `kept` given, the directory must contain exactly
  `bundle.json` plus every kept name the bundle lists. A listed stored file found in the directory
  gives `kept_mismatch` ("stored file committed"), and so does a listed kept file that is missing.
- **A listed file missing from the directory:**
  - with `store` given and `offline=False`, it is resolved, then hashed and scrub-scanned like a
    local file. A failure gives `unavailable`, `hash_mismatch` or `withdrawn`, with the detail from
    the error;
  - with `offline=True`, it is skipped if it isn't kept;
  - otherwise it is `missing_file`, as today.
- With `kept=None` and `store=None`, behavior is unchanged: every existing test passes as is.

- [ ] **Step 1: Write the failing tests**

```python
def test_split_bundle_verifies_through_store(tmp_path, fake_store):
    b = make_bundle(...); push_all(fake_store, b); split(b, KEPT_RUN)    # helpers in the test
    assert verify_bundle(b, None, None, kept=KEPT_RUN, store=Store.from_env(tmp_path)) == []

def test_stored_file_committed_is_kept_mismatch(...):   # copy quality/x.jsonl back → kind kept_mismatch
def test_offline_checks_ledger_only(...):               # no store, 0 requests, [] for a good split bundle
def test_missing_object_is_unavailable(...):            # kind "unavailable", detail has the URL
def test_withdrawn_object_reported(...):                # kind "withdrawn", detail has date and reason
def test_scrub_scan_covers_resolved_files(...):         # a private IP inside a stored output is found  # scrub-ok
def test_load_run_reads_items_from_store(...):          # split bundle → same RunView items as the complete one
def test_materialize_round_trip(...):                   # materialize(split) passes verify_bundle(kept=None)
def test_materialize_two_files_one_hash(...):           # two empty listed files, one object: both written
```

- [ ] **Step 2:** Run `uv run pytest tests/test_bundle.py tests/test_runview.py tests/test_store.py -q`.
  Expected: the new tests FAIL and the existing ones pass.
- [ ] **Step 3:** Implement per Behavior. `_read_items` takes a path from `resolve_file`. `load_run`
  passes `store` through. `materialize` copies the directory, then writes each missing listed file
  from `store.resolve`.
- [ ] **Step 4:** Run `uv run pytest -q` (the full suite). Expected: PASS.
- [ ] **Step 5:** Commit with `bundle: verify and load split bundles through the store`.

### Task 4: Signature V4 and the S3 client

**Files:**
- Create: `src/bakeoff/s3.py`, `tests/test_s3.py`
- Modify: `tests/fakestore.py` (record `PUT` bodies; respond to `HEAD` on `/<bucket>/<key>`)

**Interfaces:**
- Produces:
  - `sign(method: str, url: str, headers: dict[str, str], payload_sha256: str, *, key_id: str, secret: str, region: str, service: str, amz_date: str) -> dict[str, str]`.
    It returns the headers plus `x-amz-date`, `x-amz-content-sha256` and `Authorization`.
  - `@dataclass class S3Client: endpoint: str; bucket: str; key_id: str; secret: str`.
  - `S3Client.from_env() -> S3Client`. It raises `StoreError` naming the missing variable, but never
    a value.
  - `head(self, key: str) -> bool`.
  - `put(self, key: str, path: Path, sha256: str) -> None`.
  - `__repr__` hides the secret.

- [ ] **Step 1: Write the failing tests**

```python
def test_sigv4_known_answer():
    # AWS S3 docs example, "Signature Calculations for the Authorization Header: Transferring
    # Payload in a Single Chunk", GET Object: bucket examplebucket, key test.txt, Range bytes=0-9,
    # key AKIAIOSFODNN7EXAMPLE, secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY,
    # 20130524T000000Z, us-east-1, s3. Copy the expected Signature from that page into the test,
    # and confirm it there rather than from memory.
    h = sign("GET", "https://examplebucket.s3.amazonaws.com/test.txt", {"range": "bytes=0-9"},
             EMPTY_SHA, key_id=..., secret=..., region="us-east-1", service="s3",
             amz_date="20130524T000000Z")
    assert h["Authorization"].endswith("Signature=" + EXPECTED)

def test_put_sends_signed_immutable_object(fake_store, monkeypatch):
    # headers: Content-Type application/octet-stream, Cache-Control "public, max-age=31536000, immutable",
    # x-amz-content-sha256 == the file's sha, Authorization starts "AWS4-HMAC-SHA256 Credential=<id>/"
    # and contains "/auto/s3/aws4_request"
def test_head_true_false(fake_store, monkeypatch):     # 200 → True, 404 → False, 403 → StoreError
def test_from_env_missing_names_variable(monkeypatch): # message names BAKEOFF_STORE_SECRET, no value
def test_secret_never_in_repr_or_errors(fake_store, monkeypatch)
def test_no_delete_method():
    assert not any(n.lower().startswith("del") for n in dir(S3Client))
```

- [ ] **Step 2:** Run `uv run pytest tests/test_s3.py -q`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `s3.py` with `hmac`, `hashlib` and `urllib.request`.
  - The canonical request uses lowercase sorted signed headers and URI-encoded path segments; keys
    are hex and slashes only.
  - `put` streams the file with `Content-Length` and the global headers. It retries as in Global
    Constraints.
- [ ] **Step 4:** Run `uv run pytest tests/test_s3.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit with `s3: stdlib SigV4 client with put and head only`.

### Task 5: `bakeoff store` commands and the Reproduce line

**Files:**
- Modify: `src/bakeoff/cli.py`, `site_src/method.html`, `site/method/index.html` (regenerated)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Store`, `materialize` (Tasks 2 and 3); `S3Client` (Task 4); `verify_bundle` (Task 3).
- Produces these CLI commands, each with exit 0 on success and 1 on any problem:
  - `bakeoff store fetch BUNDLE... [--into DIR]`;
  - `bakeoff store push BUNDLE...`;
  - `bakeoff store check BUNDLE... [--rehash N]`;
  - `bakeoff verify DIR` now detects a split bundle under a split range (through `week_of` and
    `load_layout`) and verifies it with `kept` and the store.

**Output contract:**
- `fetch`: `fetched N, cached M` per bundle.
- `push`: first a full `verify_bundle(kept=None)` on each complete bundle, refusing on any finding.
  Then `uploaded N, skipped M` per bundle.
- `check`: `ok N` or one line per missing object (`missing <sha> <rel>`); a withdrawn object prints
  `withdrawn <sha> <date>`, and that alone doesn't fail the command. `--rehash N` picks N listed
  objects with `random.Random(<bundle_sha256>)`, so a failure can be reproduced.

- [ ] **Step 1: Write the failing tests:**
  - `test_store_push_then_check_then_fetch(fake_store)`: a full round trip with two bundles.
  - `test_push_refuses_bad_bundle`.
  - `test_push_is_idempotent`: the second run reports `uploaded 0`.
  - `test_check_reports_missing_and_withdrawn`.
  - `test_fetch_into_writes_complete_copy`, which passes `bakeoff verify`.
  - `test_rehash_is_deterministic`.
  - `test_push_output_never_contains_secret`.
- [ ] **Step 2:** Run `uv run pytest tests/test_cli.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement the `store` subparser, following `_parser()`'s style.
- [ ] **Step 4:** Add the Reproduce sentence to `site_src/method.html`, after the `rederive`
  paragraph: "Each split week keeps its large files outside git, by hash.
  <code>uv run bakeoff store fetch results/&lt;week&gt;/runs/&lt;id&gt;</code> downloads them and
  checks every hash against the bundle." Then run `uv run bakeoff site`.
- [ ] **Step 5:** Run `uv run pytest -q && uv run bakeoff site --check`. Expected: PASS.
- [ ] **Step 6:** Commit with `cli: bakeoff store fetch, push and check`.

### Task 6: Split-aware publish

**Files:**
- Modify: `src/bakeoff/publish.py` (`_verify`, `_stage_bundle`, `publish`)
- Test: `tests/test_publish.py`

**Interfaces:**
- Consumes: `load_layout`, `bundle_kind` (Task 1); `Store`, `verify_bundle(kept=…, store=…)`
  (Tasks 2 and 3).
- Produces: `publish(week_toml, root, *, username, denylist, username_skip_output=False, store: Store | None = None) -> Path`.
  When `store` is `None` it uses `Store.from_env(root)`, created only for a split week.

**Behavior for a week under a split range:**
- Input bundles are verified with `kept=None` and the store, so any file missing locally is resolved.
- `_verify` tolerates a `withdrawn` finding only for a file that `load_run` doesn't read
  (anything outside `quality/`). A withdrawn quality item file stays fatal.
- Then, for every input bundle, every listed file that isn't withdrawn must exist in the store
  (`Store.exists`). Otherwise raise
  `PublishError("run <id>: N object(s) not in the store; run bakeoff store push first")`.
- `_stage_bundle` copies only `bundle.json` and the listed kept files, then re-verifies the staged
  copy with `kept` and `offline=True`.
- A correction's `_published_copy` source is split: verify it with `kept` and `store`.

Weeks under `keep = "all"` behave exactly as today.

- [ ] **Step 1: Write the failing tests:**
  - `test_split_week_stages_only_kept_files(fake_store)`: the staged directories hold exactly the kept
    names, and `week.json` is byte-identical to the same week published with `keep = "all"`.
  - `test_split_week_refuses_unpushed_objects`.
  - `test_withdrawn_samples_tolerated_withdrawn_items_fatal`.
  - `test_split_judge_bundle_keeps_only_bundle_json`.
  - `test_split_correction_uses_published_copy`.
  - `test_all_range_unchanged`: the existing fixtures with no `layout.toml`, plus one with only the
    `keep = "all"` range, give identical outputs.
- [ ] **Step 2:** Run `uv run pytest tests/test_publish.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement per Behavior.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit with `publish: stage split weeks and require their objects in the store`.

### Task 7: Split-aware rederive

**Files:**
- Modify: `src/bakeoff/rederive.py` (`_copy_bundle`, `_remapped_week_toml`, `rederive`)
- Test: `tests/test_rederive.py`

**Interfaces:**
- Consumes: `materialize`, `Store`, `Withdrawn` (Tasks 2 and 3); `publish(store=…)` (Task 6).
- Produces: `rederive(root: Path, week: str, store: Store | None = None) -> list[str]`, with the same
  return contract as today.

**Behavior:**
- For a split week, each published bundle is materialized into the temporary root instead of being
  copied. `results/layout.toml` and `results/withdrawn.toml` are copied into the temporary root's
  `results/`.
- If a quality item file is withdrawn, rederive raises
  `RederiveError("<rel> in <bundle> was withdrawn on <date>: <reason>; this week can no longer be re-derived")`.
- A withdrawn telemetry or judge file is tolerated:
  - `materialize` leaves it out;
  - the temporary publish passes a `Store` built from the copied `withdrawn.toml`, so verification
    reports `withdrawn`, which Task 6 tolerates outside `quality/`;
  - the existence check skips withdrawn hashes.

- [ ] **Step 1: Write the failing tests:**
  - `test_split_week_rederives_identical(fake_store)`.
  - `test_second_rederive_offline`: the server is stopped and the cache is warm.
  - `test_withdrawn_quality_file_stops_with_reason`.
  - `test_withdrawn_samples_file_still_identical`.
  - `test_published_2026_11_02_still_identical`: runs against the repository root.
- [ ] **Step 2:** Run `uv run pytest tests/test_rederive.py -q`. Expected: the new tests FAIL, and
  `test_published_2026_11_02_still_identical` passes already.
- [ ] **Step 3:** Implement per Behavior.
- [ ] **Step 4:** Run `uv run pytest -q`, then `uv run bakeoff rederive 2026-11-02`. Expected: PASS
  and `identical`.
- [ ] **Step 5:** Commit with `rederive: materialize split weeks from the store`.

### Task 8: Ledger mode, changed mode and CI

**Files:**
- Create: `src/bakeoff/ledgercheck.py`, `.github/workflows/store-audit.yml`, `tests/test_ledgercheck.py`
- Modify: `src/bakeoff/cli.py` (`verify --ledger`, `verify --changed`), `.github/workflows/ci.yml`
- Test: `tests/test_ledgercheck.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `check_ledger(root: Path, *, username: str | None, denylist: Path | None) -> list[tuple[str, Finding]]`,
    pairing a bundle path with each finding.
  - It covers every `results/*/runs/*`, `results/*/judge/*` and `results/experiments/*/runs/*` bundle.
  - It applies the right `kept` to each bundle, with `offline=True`. It validates
    `withdrawn.toml` (every sha listed by the named bundles), and validates `layout.toml`.
  - `changed_bundles(root: Path, base: str) -> list[Path]`. It runs
    `git diff --name-only <base>...HEAD -- results/` and returns the bundle directories containing a
    changed path. It raises `LedgerError` if git can't resolve `base` (for example, a shallow
    clone).
- CLI:
  - `bakeoff verify --ledger [--root .]` is offline and never touches the network.
  - `bakeoff verify --changed BASE [--root .]` fully verifies each changed bundle with the store.
  - Both keep `--no-username` and `--denylist`. `verify DIR`, `--ledger` and `--changed` are mutually
    exclusive.

- [ ] **Step 1: Write the failing tests:**
  - `test_ledger_clean_repo`: `check_ledger(REPO, username=None, denylist=None) == []`.
  - `test_ledger_flags_committed_stored_file`.
  - `test_ledger_flags_bad_withdrawn_row`.
  - `test_ledger_makes_no_requests(fake_store)`.
  - `test_changed_bundles_in_git_repo(tmp_path)`: init a repository, commit, change one bundle, and
    expect exactly that directory.
  - `test_changed_without_base_fails_loudly`.
  - `test_cli_modes_exclusive`.
- [ ] **Step 2:** Run `uv run pytest tests/test_ledgercheck.py tests/test_cli.py -q`. Expected: the new
  tests FAIL.
- [ ] **Step 3:** Implement. In `ci.yml`:
  - give the checkout step `with: fetch-depth: 0`;
  - replace the "Verify published bundles" loop with `uv run bakeoff verify --ledger --no-username`;
  - add `uv run bakeoff verify --changed origin/${{ github.base_ref || 'main' }} --no-username`,
    with `if: github.event_name == 'pull_request'`.
- [ ] **Step 4:** Create `store-audit.yml`:
  - triggers: `schedule: cron "0 14 * * 2"` (Tuesday) and `workflow_dispatch`;
  - permissions: `contents: read`;
  - the same pinned `actions/checkout` and `setup-uv` SHAs as `ci.yml`;
  - steps: `uv sync --locked`, then a step that runs `bakeoff store check --rehash 50` over every
    bundle of every week that `load_layout` puts in a split range. With no split weeks yet, it
    exits 0 and prints `no split weeks`.
- [ ] **Step 5:** Run all of `ci.yml`'s steps locally (ruff check, ruff format --check, pytest,
  `site --check`, `verify --ledger`, `scripts/scan_repo.py`). Expected: all PASS.
- [ ] **Step 6:** Commit with `ci: offline ledger check, changed-bundle verify, weekly store audit`.

### Task 9: Yearly archive

This task is needed by 2027-03-01. It can merge after Tasks 1–8 have shipped.

**Files:**
- Create: `src/bakeoff/archive.py`, `scripts/zenodo_upload.py`, `.github/workflows/archive.yml`,
  `tests/test_archive.py`
- Modify: `src/bakeoff/cli.py` (`archive build`, `store fetch --archive`), `src/bakeoff/store.py`
  (`Store.load_archive`)

**Interfaces:**
- Consumes: `load_layout`, `load_withdrawn`, `week_of` (Task 1); `Store` (Task 2).
- Produces:
  - `build_archive(root: Path, year: int, out: Path, store: Store) -> Path`. It writes
    `MANIFEST.sha256`, `README.md` and `<year>-<MM>.tar`, one per month that has split weeks, into
    `out`, and returns `out`.
    - Manifest lines: `<sha256>  <size>  <bundle path>[,<bundle path>…]`, sorted by sha.
    - The objects are those listed by that year's split weeks, minus withdrawn ones, each stored once
      at its object key.
  - `Store.load_archive(self, archive_dir: Path) -> int`. It checks every tar member against the
    manifest, copies the matching ones into the cache, and returns the count.
  - `results/archive.toml` rows: `year`, `doi`, `manifest_sha256`.
  - `scripts/zenodo_upload.py OUT_DIR --year Y` uses the Zenodo REST API (create a deposition, `PUT`
    each file to its bucket link, set the metadata, publish) and prints the DOI. Metadata:
    - title "bakeoff.si raw data, <year>";
    - license `cc-by-4.0`;
    - community `bakeoff-si`;
    - a related identifier pointing at the previous year's DOI, if `archive.toml` has one.
  - `archive.yml`: `workflow_dispatch` with a `year` input; `environment: store`; secret
    `ZENODO_TOKEN`. It builds, uploads, then opens a pull request adding the `archive.toml` row.

- [ ] **Step 1: Write the failing tests:**
  - `test_archive_excludes_withdrawn`.
  - `test_one_tar_per_month_and_under_100_files`.
  - `test_manifest_round_trip_into_cache`: `load_archive` followed by `Store.cached` hits.
  - `test_tampered_tar_member_refused`.
  - `test_zenodo_upload_against_fake(fake_store)`: the fake answers the deposition calls, and the test
    asserts the call order and the metadata fields.
- [ ] **Step 2:** Run `uv run pytest tests/test_archive.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Check the Zenodo REST API's deposition endpoints and token scopes
  (`deposit:write`, `deposit:actions`) against developers.zenodo.org before writing the client.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit with `archive: yearly manifest and tars, Zenodo upload, fetch --archive`.

---

## After the plan (not code tasks)

The spec's rollout section covers these:
1. **Owner setup:** the R2 bucket, `data.bakeoff.si`, the cache rule, the scoped token, and the
   `store` environment.
2. **Athena:** run `bakeoff store push` before publish, and bump `bakeoff_ref`.
3. **The dry run** on a private branch.
4. **A pull request adding the `2026-11-30` split range.**
5. **First split Monday checks.**
