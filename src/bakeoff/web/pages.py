"""Page templates: pure functions from week documents and site config to HTML strings.

Every interpolated value goes through ``_e`` (``html.escape(str(v), quote=True)``). Pages carry no
``<script>``, no ``<style>`` and no ``style=`` attribute: the stylesheet is ``/style.css``.
"""

from html import escape

from bakeoff.web import fmt
from bakeoff.weekcfg import SiteConfig

DESCRIPTION = "Weekly open-model bake-offs on consumer GPUs. Open method, open data."

# Machine reason codes -> short human phrases. Unknown codes fall back to the code itself.
REASONS = {
    "ci_includes_zero": "Difference could be zero (CI includes 0)",
    "not_better": "Not better than the pick",
    "slower_than_ratio": "Decodes too slowly compared with the pick",
    "less_context": "Less usable context than the pick",
    "suite_set_mismatch": "Suites differ from the pick's",
    "suite_hash_mismatch": "Suite hashes differ from the pick's",
    "item_ids_mismatch": "Test items differ from the pick's",
    "pick_not_remeasured": "Pick not re-measured this week",
    "status_incomplete": "Run did not complete",
    "no_fit": "No context-fit result",
    "context_below_min": "Usable context below the minimum",
    "no_speed": "No speed result",
    "decode_below_min": "Decode speed below the minimum",
}

VERDICTS = {
    "replace": "Replace",
    "hold": "Hold",
    "watch": "Watch",
    "initial_pick": "Initial pick",
}
_BADGE_CLASSES = ("replace", "hold", "watch", "initial_pick")
_SPEED_CELL = "short@1"
_LOGO = '<img src="/logo.webp" width="440" height="360" alt="bakeoff.si">'
_MAKER = (
    '<a class="maker" href="https://www.atlantisanalytics.com/">'
    '<img src="/atlantis.webp" width="40" height="40" alt="">A project of Atlantis Analytics</a>'
)
_LEDE = (
    "Every week, new open-weight models get tested on ordinary consumer GPUs, "
    "and the results are published Monday morning."
)


def _e(v: object) -> str:
    return escape(str(v), quote=True)


def _link(href: object, text: str) -> str:
    """An anchor for https:// and site-relative hrefs only; anything else is plain text.

    ``text`` must already be escaped.
    """
    h = str(href)
    if not (h.startswith("https://") or (h.startswith("/") and not h.startswith("//"))):
        return text
    return f'<a href="{_e(h)}">{text}</a>'


def reason_text(code: object) -> str:
    return REASONS.get(code, str(code)) if isinstance(code, str) else str(code)


def _reasons(codes: list) -> str:
    return "; ".join(_e(reason_text(c)) for c in codes)


# --- layout ---------------------------------------------------------------------------------


def _head(title: str, path: str, site: SiteConfig) -> str:
    url = _e(site.base_url.rstrip("/") + path)
    t = _e(title)
    return f"""<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="theme-color" content="#fefefe">
<title>{t}</title>
<meta name="description" content="{_e(DESCRIPTION)}">
<link rel="canonical" href="{url}">
<link rel="stylesheet" href="/style.css">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<meta property="og:type" content="website">
<meta property="og:url" content="{url}">
<meta property="og:title" content="{t}">
<meta property="og:description" content="{_e(DESCRIPTION)}">
<meta property="og:image" content="{_e(site.base_url.rstrip("/"))}/og.jpg">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
</head>"""


def newsletter_form(site: SiteConfig) -> str:
    if not site.newsletter_action:
        return ""
    return f"""<form class="signup" method="post" action="{_e(site.newsletter_action)}">
<label for="signup-email">Get each week's results by email</label>
<input type="email" id="signup-email" name="{_e(site.email_field)}" required autocomplete="email">
<button type="submit">Subscribe</button>
<a href="/privacy/">Privacy</a>
</form>"""


def _footer(site: SiteConfig, week: str | None = None, lead: str = "") -> str:
    links = ['<a href="/method/">Method</a>']
    if site.launched:
        links.append('<a href="/weeks/">Archive</a>')
    if week is not None:
        links.append(f'<a href="/api/v1/weeks/{_e(week)}.json">This week as JSON</a>')
    if site.launched:
        links.append('<a href="/feed.xml">Feed</a>')
    links.append('<a href="/privacy/">Privacy</a>')
    form = newsletter_form(site)
    return f"""<div class="site-footer">
<footer>
{lead}<p class="footer-links">{"".join(links)}</p>
{form}
<p>{_MAKER}</p>
</footer>
</div>"""


def _layout(title: str, path: str, site: SiteConfig, main: str, *, week: str | None = None) -> str:
    return f"""<!doctype html>
<html lang="en">
{_head(title, path, site)}
<body>
<div class="site-header"><a href="/">{_LOGO}</a></div>
<main>
{main}
</main>
{_footer(site, week)}
</body>
</html>
"""


def wrap_static(fragment: str, title: str, site: SiteConfig, path: str = "/") -> str:
    """Wrap a hand-written body fragment (method, privacy) in the shared layout."""
    return _layout(title, path, site, fragment)


def holding_page(site: SiteConfig) -> str:
    lead = "<strong>First results soon.</strong>\n"
    return f"""<!doctype html>
<html lang="en">
{_head("bakeoff.si", "/", site)}
<body class="holding">
<div>
<main>
  <h1>{_LOGO}</h1>
  <p class="lede">{_LEDE}</p>
  <ul>
    <li>Which model is best for <em>your</em> card: 12&nbsp;GB, 24&nbsp;GB, 36&nbsp;GB unified</li>
    <li>Quality, tokens/s, time to first token, VRAM, usable context, license</li>
    <li>Open method, raw data, reproducible runs</li>
  </ul>
</main>
{_footer(site, lead=lead)}
</div>
</body>
</html>
"""


# --- week page ------------------------------------------------------------------------------


def _bundle_link(run: dict, site: SiteConfig) -> str:
    href = f"{site.repo_url.rstrip('/')}/tree/main/{run['bundle_path']}"
    return _link(href, "Raw bundle")


def _model_cell(run: dict) -> str:
    m = run["model"]
    label = _e(m.get("label"))
    out = [f'<strong title="{label}">{label}</strong>']
    out.append(f'<div class="small muted">{_e(m.get("quant"))} · {_e(m.get("runtime"))}</div>')
    lic = _e(m.get("license"))
    if m.get("license_url"):
        lic = _link(m["license_url"], lic)
    src = m.get("source_url")
    src_html = f" · {_link(src, 'Source')}" if src else ""
    out.append(f'<div class="small">{lic}{src_html}</div>')
    return "".join(out)


def _suite_line(s: dict, track: str) -> str:
    name = _e(str(s.get("key", "")).split("/", 1)[-1])
    strict = f"{fmt.pct(s.get('accuracy'))} {fmt.pm(s.get('ci95'))}"
    if track == "structured":
        if s.get("extractable_accuracy") is not None:
            ext = f"{fmt.pct(s['extractable_accuracy'])} {fmt.pm(s.get('extractable_ci95'))}"
        else:
            ext = "not measured"
        body = f"strict {_e(strict)} · extractable {_e(ext)}"
    else:
        body = _e(strict)
    return f'<div class="small muted">{name}: {body}</div>'


def _quality_cell(run: dict) -> str:
    tracks = run["quality"]["tracks"]
    if not tracks:
        return fmt.DASH
    parts = []
    for track in sorted(tracks):
        t = tracks[track]
        lines = "".join(_suite_line(s, track) for s in t.get("suites", []))
        parts.append(
            f'<div class="block"><span class="track">{_e(track)}</span> '
            f"<strong>{_e(fmt.pct(t.get('score')))}</strong>{lines}</div>"
        )
    return "".join(parts)


def _verdict_cell(run: dict, label_of: dict) -> str:
    parts = []
    if run["role"] == "pick":
        tracks = ", ".join(_e(t) for t in run["pick_tracks"])
        parts.append(
            f'<div class="block"><span class="badge badge-pick">Current pick</span> '
            f'<span class="track">{tracks}</span></div>'
        )
    for track in sorted(run["verdicts"]):
        parts.append(_one_verdict(track, run["verdicts"][track], label_of))
    return "".join(parts) or fmt.DASH


def _one_verdict(track: str, v: dict, label_of: dict) -> str:
    final = v["final"]
    if final == "replace" and not v.get("new_pick") and not v.get("outranked_by"):
        badge, cls = "same model as pick", "same"
    else:
        badge = VERDICTS.get(final, final)
        cls = final if final in _BADGE_CLASSES else "hold"
    out = [
        f'<span class="track">{_e(track)}</span> <span class="badge badge-{cls}">{_e(badge)}</span>'
    ]
    if v.get("override_reason") is not None:
        out.append(
            f'<div class="small">Computed: {_e(v["computed"])}, owner: {_e(final)}. '
            f"{_e(v['override_reason'])}</div>"
        )
    if v.get("new_pick"):
        note = "Now the pick" if final == "initial_pick" else "Becomes the pick next week"
        out.append(f'<div class="small">{note}</div>')
    if v.get("outranked_by"):
        by = v["outranked_by"]
        out.append(f'<div class="small">Outranked by {_e(label_of.get(by) or by)}</div>')
    if v.get("diff") is None or v.get("diff_ci95") is None:
        diff = f"Difference {fmt.DASH}"
    else:
        lo, hi = v["diff_ci95"]
        diff = f"{fmt.pts(v['diff'])} pts (CI {fmt.pts(lo)} to {fmt.pts(hi)})"
    out.append(f'<div class="small muted">{_e(diff)}</div>')
    if v.get("reasons"):
        out.append(f'<div class="small muted">{_reasons(v["reasons"])}</div>')
    return f'<div class="block">{"".join(out)}</div>'


def _kv_line(fit: dict) -> str:
    """Under the context figure: the KV pool, and the share one max-size request needs."""
    capacity = fit.get("kv_capacity_tokens")
    if capacity is None:
        return ""
    text = f"KV pool {fmt.kctx(capacity)}"
    if fit.get("kv_fraction_at_max") is not None:
        text += f" · {fmt.pct(fit['kv_fraction_at_max'])} at max"
    return f'<div class="small muted">{_e(text)}</div>'


def _speed_cells(run: dict) -> str:
    cell = run["speed"].get(_SPEED_CELL, {})
    decode = (cell.get("decode_tps") or {}).get("median")
    ttft = (cell.get("ttft_s") or {}).get("median")
    ctx = run["fit"].get("max_context_passed")
    vram = run["memory"].get("size_vram")
    cells = [fmt.tps(decode), fmt.ms(ttft), fmt.kctx(ctx), fmt.gb(vram)]
    extra = ["", "", _kv_line(run["fit"]), ""]
    return "".join(f'<td class="num">{_e(v)}{x}</td>' for v, x in zip(cells, extra, strict=True))


def _run_row(run: dict, site: SiteConfig, label_of: dict) -> str:
    cls = ' class="pick"' if run["role"] == "pick" else ""
    return (
        f"<tr{cls}><td>{_model_cell(run)}</td><td>{_quality_cell(run)}</td>"
        f"{_speed_cells(run)}<td>{_verdict_cell(run, label_of)}</td>"
        f"<td>{_bundle_link(run, site)}</td></tr>"
    )


_HEAD_ROW = (
    "<tr><th>Model</th><th>Quality</th>"
    '<th class="num">Decode tok/s</th><th class="num">TTFT</th>'
    '<th class="num">Context</th><th class="num">VRAM</th><th>Verdict</th><th>Data</th></tr>'
)


def _pick_status(status: dict, picks: dict, week: str) -> str:
    """One note per track whose pick was not re-measured, or that had no pick before this week.

    ``picks`` is the tier's ``picks_after``: a pick that was not re-measured carries forward
    unchanged, so it names the current pick. Should an override have replaced it anyway, its
    ``since_week`` is this week and only the previous pick's status is stated.
    """
    notes = []
    for track in sorted(status):
        s = status[track]
        p = picks.get(track) if isinstance(picks.get(track), dict) else {}
        if s == "not_re-measured":
            if p and p.get("since_week") != week:
                label = p.get("label") or p.get("run_id")
                notes.append(
                    f"{track}: Current pick: {label} (since {p.get('since_week')}) "
                    "— not re-measured this week"
                )
            else:
                notes.append(f"{track}: the previous pick was not re-measured this week")
        elif s == "none":
            notes.append(f"{track}: No previous pick — initial pick chosen this week")
    return "\n".join(f'<p class="small muted">{_e(n)}.</p>' for n in notes)


def _model_of(d: dict) -> tuple:
    return (d.get("weights"), d.get("quant"), d.get("runtime"))


def _misfit_item(r: dict, site: SiteConfig, pick_tracks: list[str]) -> str:
    label = _e(r["model"].get("label"))
    mark = ""
    if pick_tracks:
        tracks = ", ".join(_e(t) for t in pick_tracks)
        mark = (
            f' <span class="badge badge-pick">Current pick</span> <span class="track">{tracks}'
            "</span>, not re-measured"
        )
    return (
        f'<li><strong title="{label}">{label}</strong> ({_e(r["run_id"])}){mark}: '
        f"{_reasons(r['ineligible_reasons'])}. {_bundle_link(r, site)}</li>"
    )


def _tier_section(tier: dict, site: SiteConfig, picks: dict, week: str) -> str:
    runs = tier["runs"]
    status = tier.get("pick_status", {})
    label_of = {r["run_id"]: r["model"].get("label") for r in runs}
    fits = sorted(
        (r for r in runs if r["eligible"]), key=lambda r: (r["role"] != "pick", r["run_id"])
    )
    misfits = [r for r in runs if not r["eligible"]]
    out = [
        f'<section class="tier" id="tier-{_e(tier["id"])}">',
        f"<h2>{_e(tier['name'])}</h2>",
        f'<p class="small muted">{_e(tier["envelope"])}</p>',
        _pick_status(status, picks, week),
    ]
    if fits:
        rows = "".join(_run_row(r, site, label_of) for r in fits)
        out.append(f'<div class="table-wrap"><table>{_HEAD_ROW}{rows}</table></div>')
    else:
        out.append("<p>No run fit this tier this week.</p>")
    if misfits:
        out.append("<h3>Tested, doesn&#x27;t fit</h3>")
        stale = {
            t: _model_of(picks[t])
            for t, s in status.items()
            if s == "not_re-measured" and isinstance(picks.get(t), dict)
        }
        items = "".join(
            _misfit_item(r, site, [t for t in sorted(stale) if stale[t] == _model_of(r["model"])])
            for r in misfits
        )
        out.append(f'<ul class="flat">{items}</ul>')
    if tier.get("judge"):
        items = "".join(
            f"<li>Judge: {_e(j['wins'])} wins, {_e(j['ties'])} ties, {_e(j['losses'])} losses vs "
            f"{_e(label_of.get(j['b'], j['b']))} (record for {_e(label_of.get(j['a'], j['a']))}; "
            f"judge model {_e(j['judge_model'])}, {_e(j['errors'])} errors). Information only.</li>"
            for j in tier["judge"]
        )
        out.append(f'<h3>Judge results</h3><ul class="flat">{items}</ul>')
    out.append("</section>")
    return "\n".join(p for p in out if p)


_COUNT_WORDS = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
_SUITE_NAMES = {"gsm8k": "GSM8K", "ifeval": "IFEval", "json": "JSON"}  # others show their slug


def _agreement_section(doc: dict) -> str:
    """One line per model measured on several cards: how often their pass/fail agreed."""
    entries = doc.get("quality_agreement") or []
    if not entries:
        return ""
    label_of = {r["run_id"]: r["model"]["label"] for t in doc["tiers"] for r in t["runs"]}
    items = []
    for a in entries:
        label = next((label_of[r] for r in a["runs"] if r in label_of), a["weights"])
        parts = [
            f"{round(100 * c['same_pass'] / c['n'])}% "
            f"({_SUITE_NAMES.get(key.split('/', 1)[-1], key.split('/', 1)[-1])})"
            for key, c in sorted(a["suites"].items())
            if c["n"]
        ]
        if not parts:
            continue
        agreed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        items.append(
            f"<li>{_e(label)}: same model on "
            f"{_COUNT_WORDS.get(len(a['runs']), len(a['runs']))} cards, "
            f"pass/fail agreed on {_e(agreed)} of items.</li>"
        )
    if not items:
        return ""
    return f'<h2>Same model, several cards</h2>\n<ul class="flat">{"".join(items)}</ul>'


def week_page(doc: dict, site: SiteConfig) -> str:
    week = doc["week"]
    rules = doc.get("rules", {})
    meta = f"Published {_e(doc['published_at'])}, revision {_e(doc['revision'])}."
    corr = ""
    if doc.get("corrections"):
        items = "".join(
            f"<li>Revision {_e(c['revision'])} ({_e(c['published_at'])}): {_e(c['note'])}</li>"
            for c in doc["corrections"]
        )
        corr = f'<h3>Corrections</h3><ul class="flat">{items}</ul>'
    rules_line = (
        f"Rules v{_e(rules.get('version'))}: "
        f"minimum context {_e(fmt.kctx(rules.get('min_context')))}, "
        f"minimum decode {_e(fmt.tps(rules.get('min_decode_tps')))} tok/s, "
        f"speed ratio {_e(rules.get('speed_ratio'))}. "
        '<a href="/method/">How verdicts work</a>.'
    )
    picks = doc.get("picks_after", {})
    sections = "\n".join(_tier_section(t, site, picks.get(t["id"], {}), week) for t in doc["tiers"])
    main = (
        f'<h1 class="title">Week of {_e(week)}</h1>\n<p class="muted">{meta}</p>\n'
        f'<p class="small muted">{rules_line}</p>\n{corr}\n{sections}{_agreement_section(doc)}'
    )
    return _layout(f"bakeoff.si, week of {week}", f"/weeks/{week}/", site, main, week=week)


def archive_page(index: list[dict], site: SiteConfig) -> str:
    if index:
        items = "".join(
            f"<li>{_link(w['url'], 'Week of ' + _e(w['week']))} "
            f'<span class="muted">revision {_e(w["revision"])}, {_e(w["published_at"])}</span></li>'
            for w in index
        )
        body = f'<ul class="flat">{items}</ul>'
    else:
        body = "<p>No weeks published yet.</p>"
    main = f'<h1 class="title">Archive</h1>\n{body}'
    return _layout("bakeoff.si, archive", "/weeks/", site, main)
