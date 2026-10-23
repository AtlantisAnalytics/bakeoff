import copy

import pytest
from htmlcheck import assert_balanced
from weekfix import make_bundle

from bakeoff.runview import load_run
from bakeoff.verdicts import RULES_V1, assess_tier
from bakeoff.web import fmt
from bakeoff.web.pages import (
    archive_page,
    holding_page,
    newsletter_form,
    week_page,
    wrap_static,
)
from bakeoff.week import build_week
from bakeoff.weekcfg import RunEntry, SiteConfig, Tier, WeekInput

RULES = {**RULES_V1, "resamples": 200}
WEEK = "2026-10-19"
AT = "2026-10-19T07:00:00Z"
ENV = "lab-box-7 24 GB"
GSM = "general/gsm8k"
JSN = "structured/json"
REPO = "https://github.com/example/bakeoff"
LABEL = 'Qwen3 "Instruct" <beta> & co'

SITE = SiteConfig(
    launched=True,
    base_url="https://bakeoff.si",
    repo_url=REPO,
    newsletter_action="https://news.example.org/subscribe?a=1&b=2",
    email_field="email_field",
)
QUIET = SiteConfig(True, "https://bakeoff.si", REPO, "", "email")


def g(k, n=50):
    return [True] * k + [False] * (n - k)


def pick(track, weights):
    return {
        "tier": "24gb",
        "track": track,
        "label": weights,
        "weights": weights,
        "quant": "Q4_K_M",
        "runtime": "ollama",
        "since_week": "2026-10-12",
        "bundle_sha256": "0" * 64,
        "run_id": "old-run",
    }


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("web")

    def mk(rid, weights, grades, label=None, fmt="m2", **kw):
        path = make_bundle(
            tmp,
            label=label or weights,
            weights=weights,
            envelope=ENV,
            grades=grades,
            run_id=rid,
            fmt=fmt,
            **kw,
        )
        entry = RunEntry(path, "24gb", "Apache-2.0", "https://example.org/" + weights, None)
        return load_run(path), entry

    made = [
        mk("r-ctrl", "base", {GSM: g(20), JSN: g(30)}),
        mk("r-cand", "big", {GSM: g(48), JSN: g(45)}, label=LABEL),
        mk("r-old", "oldfmt", {GSM: g(10), JSN: g(10)}, fmt="m1"),
        mk("r-slow", "slow", {GSM: g(30), JSN: g(30)}, decode=3.0),
    ]
    runs = {v.run_id: (v, e) for v, e in made}
    tiers = {"24gb": Tier("24gb", "24 GB", ENV)}
    rs = [v for v, _ in runs.values()]
    picks = {"general": pick("general", "base"), "structured": pick("structured", "base")}
    results = {"24gb": assess_tier(WEEK, "24gb", rs, picks, [], RULES)}
    inp = WeekInput(WEEK, AT, [e for _, e in made], [], [], None, b"")
    judges = [
        {
            "a": "r-ctrl",
            "b": "r-cand",
            "wins": 12,
            "ties": 3,
            "losses": 5,
            "errors": 0,
            "judge_model": "judge-x",
            "bundle_sha256": "a" * 64,
        }
    ]
    return build_week(
        inp,
        tiers,
        runs,
        results,
        judges,
        revision=1,
        published_at=AT,
        corrections=[],
        rules=RULES,
    )


def run_of(d, rid):
    return next(r for t in d["tiers"] for r in t["runs"] if r["run_id"] == rid)


# fmt


def test_fmt_golden():
    assert fmt.pct(0.87) == "87.0%"
    assert fmt.pct(1) == "100.0%"
    assert fmt.pm([0.801, 0.939]) == "±6.9"
    assert fmt.tps(47.66) == "47.7"
    assert fmt.ms(0.0424) == "42 ms"
    assert fmt.kctx(32768) == "32K"
    assert fmt.kctx(16384) == "16K"


@pytest.mark.parametrize("f", [fmt.pct, fmt.pm, fmt.tps, fmt.ms, fmt.kctx])
def test_fmt_none(f):
    assert f(None) == "—"


# holding page and form


def test_holding_page_keeps_copy_and_has_no_inline_style_or_script():
    h = holding_page(SITE)
    assert "Every week, new open-weight models get tested" in h
    assert "published Monday morning." in h
    assert h.count("<li>") == 3
    assert "Which model is best for <em>your</em> card" in h
    assert "Open method, raw data, reproducible runs" in h
    assert "First results soon." in h
    assert 'href="/style.css"' in h
    for bad in ("<style", "style=", "<script"):
        assert bad not in h
    assert_balanced(h)


def test_newsletter_form_only_with_action():
    assert newsletter_form(QUIET) == ""
    f = newsletter_form(SITE)
    assert 'method="post"' in f
    assert 'action="https://news.example.org/subscribe?a=1&amp;b=2"' in f
    assert 'name="email_field"' in f
    assert 'href="/privacy/"' in f
    assert_balanced(f)
    assert "<form" not in holding_page(QUIET)
    assert "<form" in holding_page(SITE)


# week page


def test_week_page_escapes_and_structure(doc):
    h = week_page(doc, SITE)
    assert LABEL not in h
    esc = "Qwen3 &quot;Instruct&quot; &lt;beta&gt; &amp; co"
    assert f">{esc}<" in h or esc in h
    assert f'title="{esc}"' in h
    assert_balanced(h)
    for bad in ("<style", "style=", "<script"):
        assert bad not in h
    assert "24 GB" in h and "<section" in h
    assert f"{REPO}/tree/main/results/{WEEK}/runs/r-cand" in h
    assert "Tested, doesn&#x27;t fit" in h or "Tested, doesn't fit" in h
    assert "Apache-2.0" in h
    assert f"/api/v1/weeks/{WEEK}.json" in h and "/feed.xml" in h and "/method/" in h


def test_week_page_pick_and_verdicts(doc):
    h = week_page(doc, SITE)
    assert "Current pick" in h
    assert "Replace" in h
    assert "Computed:" not in h
    assert "decode_below_min" not in h  # code mapped to a phrase
    assert "Decode speed below the minimum" in h
    assert "32K" in h


def test_week_page_ineligible_reasons(doc):
    h = week_page(doc, SITE)
    assert "(r-slow): Decode speed below the minimum." in h
    assert "not re-measured" not in h  # the pick was re-measured


def test_strict_extractable_and_not_measured(doc):
    h = week_page(doc, SITE)
    assert "strict" in h and "extractable" in h
    assert "not measured" in h  # the m1 run


def test_override_line(doc):
    d = copy.deepcopy(doc)
    v = run_of(d, "r-cand")["verdicts"]["general"]
    v.update(computed="hold", final="replace", override_reason="owner saw the <logs>")
    h = week_page(d, SITE)
    assert "Computed: hold, owner: replace" in h
    assert "owner saw the &lt;logs&gt;" in h


def test_same_model_as_pick_and_outranked_and_null_diff(doc):
    d = copy.deepcopy(doc)
    v = run_of(d, "r-cand")["verdicts"]["general"]
    v.update(final="replace", new_pick=False)
    v.pop("outranked_by", None)
    assert "same model as pick" in week_page(d, SITE)
    v.update(outranked_by="r-other")
    h = week_page(d, SITE)
    assert "Outranked by r-other" in h
    v.update(diff=None, diff_ci95=None, reasons=["suite_hash_mismatch", "weird_code"])
    h = week_page(d, SITE)
    assert "Suite hashes differ" in h and "weird_code" in h
    assert_balanced(h)


def test_unknown_values_are_escaped(doc):
    d = copy.deepcopy(doc)
    run_of(d, "r-slow")["ineligible_reasons"] = ["<b>x</b>"]
    h = week_page(d, SITE)
    assert "<b>x</b>" not in h and "&lt;b&gt;x&lt;/b&gt;" in h


def test_judge_info_only(doc):
    h = week_page(doc, SITE)
    assert "Judge: 12 wins, 3 ties, 5 losses" in h
    assert "r-ctrl" in h and "r-cand" in h


def test_no_form_on_week_page_without_action(doc):
    assert "<form" not in week_page(doc, QUIET)
    assert "<form" in week_page(doc, SITE)


def test_corrections_shown(doc):
    d = copy.deepcopy(doc)
    d["revision"] = 2
    d["corrections"] = [{"revision": 2, "published_at": AT, "note": "regraded <x>"}]
    h = week_page(d, SITE)
    assert "regraded &lt;x&gt;" in h
    assert_balanced(h)


# archive and static


def test_archive_page():
    idx = [
        {"week": "2026-10-19", "revision": 1, "published_at": AT, "url": "/weeks/2026-10-19/"},
        {"week": "2026-10-12", "revision": 2, "published_at": AT, "url": "/weeks/2026-10-12/"},
    ]
    h = archive_page(idx, SITE)
    assert h.index("2026-10-19") < h.index("2026-10-12")
    assert 'href="/weeks/2026-10-19/"' in h
    assert_balanced(h)
    assert "No weeks" in archive_page([], SITE)


def test_wrap_static():
    h = wrap_static("<h2>Hi</h2><p>x</p>", "Method & more", SITE)
    assert "<h2>Hi</h2>" in h
    assert "<title>Method &amp; more" in h
    assert 'href="/style.css"' in h
    assert_balanced(h)


def test_feed_link_only_when_launched(doc):
    off = SiteConfig(False, "https://bakeoff.si", REPO, "", "email")
    assert "/feed.xml" not in wrap_static("<p>x</p>", "T", off)
    assert "/feed.xml" in wrap_static("<p>x</p>", "T", SITE)


@pytest.mark.parametrize("name", ["method", "privacy"])
def test_site_src_fragments_wrap_cleanly(name):
    from pathlib import Path

    frag = (Path(__file__).parent.parent / "site_src" / f"{name}.html").read_text(encoding="utf-8")
    h = wrap_static(frag, name, SITE)
    assert_balanced(h)
    for bad in ("<style", "style=", "<script"):
        assert bad not in h


def test_stylesheet_has_tokens_and_wrapper():
    from pathlib import Path

    css = (Path(__file__).parent.parent / "site" / "style.css").read_text(encoding="utf-8")
    for needle in ("--amber", ".table-wrap", "overflow-x: auto", ".badge-watch", "td.num"):
        assert needle in css


# pick status: not re-measured, initial picks, outranked labels


def _doc_for(tmp, specs, picks):
    """specs: (run_id, weights, grades, kw); picks: {track: pick dict}."""
    made = []
    for rid, weights, grades, kw in specs:
        path = make_bundle(
            tmp,
            label=weights.title(),
            weights=weights,
            envelope=ENV,
            grades=grades,
            run_id=rid,
            **kw,
        )
        entry = RunEntry(path, "24gb", "Apache-2.0", "https://example.org/" + weights, None)
        made.append((load_run(path), entry))
    runs = {v.run_id: (v, e) for v, e in made}
    tiers = {"24gb": Tier("24gb", "24 GB", ENV)}
    res = {"24gb": assess_tier(WEEK, "24gb", [v for v, _ in made], picks, [], RULES)}
    inp = WeekInput(WEEK, AT, [e for _, e in made], [], [], None, b"")
    return build_week(
        inp, tiers, runs, res, [], revision=1, published_at=AT, corrections=[], rules=RULES
    )


def labelled_pick(track, weights, label):
    return {**pick(track, weights), "label": label}


NOT_REMEASURED = "general: Current pick: Gone 8B (since 2026-10-12) — not re-measured this week"


def test_not_remeasured_pick_is_named_when_control_missing(tmp_path):
    d = _doc_for(
        tmp_path,
        [("r-cand", "big", {GSM: g(40)}, {})],
        {"general": labelled_pick("general", "gone", "Gone 8B")},
    )
    assert d["tiers"][0]["pick_status"] == {"general": "not_re-measured"}
    h = week_page(d, SITE)
    assert NOT_REMEASURED in h
    assert_balanced(h)


def test_not_remeasured_pick_is_named_for_tier_without_runs(tmp_path):
    d = _doc_for(tmp_path, [], {"general": labelled_pick("general", "gone", "Gone 8B")})
    h = week_page(d, SITE)
    assert NOT_REMEASURED in h
    assert "No run fit this tier this week." in h


def test_ineligible_control_is_marked_as_the_pick(tmp_path):
    d = _doc_for(
        tmp_path,
        [
            ("r-slow", "slow", {GSM: g(30)}, {"decode": 3.0}),
            ("r-cand", "big", {GSM: g(40)}, {}),
        ],
        {"general": labelled_pick("general", "slow", "Slow 8B")},
    )
    assert d["tiers"][0]["pick_status"] == {"general": "not_re-measured"}
    h = week_page(d, SITE)
    assert "general: Current pick: Slow 8B (since 2026-10-12) — not re-measured this week" in h
    misfits = h[h.index("Tested, doesn") :]
    item = misfits[: misfits.index("</li>")]
    assert "r-slow" in item and "Current pick" in item and "general" in item
    assert_balanced(h)


def test_initial_pick_wording(tmp_path):
    d = _doc_for(
        tmp_path,
        [("r-a", "alpha", {GSM: g(40)}, {}), ("r-b", "beta", {GSM: g(20)}, {})],
        {},
    )
    h = week_page(d, SITE)
    assert "general: No previous pick — initial pick chosen this week" in h
    assert "Now the pick" in h
    assert "no pick yet" not in h
    assert "Becomes the pick next week" not in h


def test_replace_new_pick_keeps_becomes_wording(doc):
    h = week_page(doc, SITE)
    assert "Becomes the pick next week" in h
    assert "Now the pick" not in h


def test_outranked_by_shows_label(doc):
    d = copy.deepcopy(doc)
    v = run_of(d, "r-cand")["verdicts"]["general"]
    v.update(final="replace", new_pick=False, outranked_by="r-ctrl")
    assert "Outranked by base" in week_page(d, SITE)


MAKER = '<a class="maker" href="https://www.atlantisanalytics.com/">'


def test_footer_credits_atlantis_on_every_page(doc):
    for h in (holding_page(QUIET), wrap_static("<p>x</p>", "T", SITE), week_page(doc, SITE)):
        assert MAKER in h
        assert '<img src="/atlantis.webp" width="40" height="40" alt="">' in h
        assert "A project of Atlantis Analytics" in h
        assert_balanced(h)


def test_atlantis_emblem_is_a_shipped_asset():
    from pathlib import Path

    from bakeoff.web.build import ASSETS

    assert "atlantis.webp" in ASSETS
    assert (Path(__file__).parent.parent / "site" / "atlantis.webp").is_file()


def _with_agreement(doc):
    d = copy.deepcopy(doc)
    first = d["tiers"][0]["runs"][0]
    d["quality_agreement"] = [
        {
            "weights": first["model"]["weights"],
            "quant": first["model"]["quant"],
            "runtime": first["model"]["runtime"],
            "runs": [first["run_id"], "r-other-tier"],
            "suites": {
                "general/gsm8k": {"n": 100, "same_pass": 90, "same_output": 32},
                "general/ifeval": {"n": 100, "same_pass": 94, "same_output": 12},
            },
        }
    ]
    return d, first["model"]["label"]


def test_week_page_agreement_line(doc):
    d, label = _with_agreement(doc)
    h = week_page(d, SITE)
    assert "Same model, several cards" in h
    assert fmt_label(label) in h
    assert "same model on two cards, pass/fail agreed on 90% (GSM8K) and 94% (IFEval) of items" in h
    assert_balanced(h)


def test_week_page_no_agreement(doc):
    assert "agreed on" not in week_page(doc, SITE)


def fmt_label(label):
    from html import escape

    return escape(label, quote=True)


def test_agreement_line_spells_small_counts_and_falls_back_to_slugs(doc):
    d, _ = _with_agreement(doc)
    d["quality_agreement"][0]["runs"] = [f"r{i}" for i in range(12)]
    d["quality_agreement"][0]["suites"] = {
        "structured/new-thing": {"n": 4, "same_pass": 3, "same_output": 3}
    }
    h = week_page(d, SITE)
    assert "same model on 12 cards, pass/fail agreed on 75% (new-thing) of items" in h


def _kv(doc, capacity, fraction):
    d = copy.deepcopy(doc)
    fit = d["tiers"][0]["runs"][0]["fit"]
    fit["kv_capacity_tokens"], fit["kv_fraction_at_max"] = capacity, fraction
    return d


def test_week_page_shows_the_kv_pool_under_context(doc):
    h = week_page(_kv(doc, 802240, 0.0408), SITE)
    assert '<div class="small muted">KV pool 783K · 4.1% at max</div>' in h
    assert_balanced(h)


def test_week_page_kv_pool_without_a_fraction(doc):
    h = week_page(_kv(doc, 802240, None), SITE)
    assert '<div class="small muted">KV pool 783K</div>' in h


def test_week_page_without_kv_fields_has_no_kv_line(doc):
    assert "KV pool" not in week_page(doc, SITE)
    assert "KV pool" not in week_page(_kv(doc, None, None), SITE)
