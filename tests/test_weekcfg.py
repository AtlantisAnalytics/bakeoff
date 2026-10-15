from pathlib import Path

import pytest

from bakeoff.weekcfg import WeekInputError, load_site, load_tiers, load_week

ROOT = Path(__file__).resolve().parent.parent

GOOD = """\
week = "2026-10-19"
published_at = "2026-10-19T07:00:00Z"

[[run]]
bundle = "runs/20261018T000000-qwen3-14b"
tier = "24gb"
license = "Apache-2.0"
source_url = "https://huggingface.co/x/y"

[[judge]]
bundle = "runs/judge-1"

[[override]]
tier = "24gb"
track = "general"
run = "20261018T000000-qwen3-14b"
verdict = "hold"
reason = "Known tokenizer bug."

[correction]
note = "Re-graded."
"""


@pytest.fixture
def tiers():
    return load_tiers(ROOT / "tiers.toml")


def write(tmp_path, text, name="week.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_shipped_files_load(tiers):
    assert list(tiers) == ["12gb", "24gb", "36gb-unified"]
    assert tiers["12gb"].envelope == "1× RTX 3060 12 GB"
    site = load_site(ROOT / "site.toml")
    assert site.launched is False
    assert site.base_url == "https://bakeoff.si"
    assert site.newsletter_action == ""
    assert site.email_field == "email"


def test_good_week(tmp_path, tiers):
    p = write(tmp_path, GOOD)
    w = load_week(p, tiers)
    assert w.week == "2026-10-19"
    assert w.published_at == "2026-10-19T07:00:00Z"
    assert w.runs[0].bundle == tmp_path / "runs/20261018T000000-qwen3-14b"
    assert w.runs[0].license_url is None
    assert w.judges == [tmp_path / "runs/judge-1"]
    assert w.overrides[0].verdict == "hold"
    assert w.correction == "Re-graded."
    assert w.source_bytes == p.read_bytes()


def test_minimal_week(tmp_path, tiers):
    text = GOOD.split("[[judge]]")[0]
    w = load_week(write(tmp_path, text), tiers)
    assert w.judges == [] and w.overrides == [] and w.correction is None


@pytest.mark.parametrize(
    "text,match",
    [
        (GOOD + "\nbogus = 1\n", "bogus"),
        (GOOD.replace('tier = "24gb"\nlicense', 'tier = "24gb"\nextra = 1\nlicense'), "extra"),
        (GOOD.replace('note = "Re-graded."', 'note = "x"\nzzz = 1'), "zzz"),
        (GOOD.replace('week = "2026-10-19"\n', ""), "week"),
        (GOOD.replace('week = "2026-10-19"', 'week = "2026-10-20"'), "week"),
        (GOOD.replace('week = "2026-10-19"', 'week = "2026-02-30"'), "week"),
        (GOOD.replace("2026-10-19T07:00:00Z", "2026-10-19T07:00:00"), "published_at"),
        (GOOD.replace("2026-10-19T07:00:00Z", "2026-10-19T07:00:00+00:00"), "published_at"),
        (GOOD.replace("2026-10-19T07:00:00Z", "2026-13-19T07:00:00Z"), "published_at"),
        (GOOD.replace('tier = "24gb"\nlicense', 'tier = "nope"\nlicense'), "tier"),
        (GOOD.replace("https://huggingface.co/x/y", "javascript:alert(1)"), "source_url"),
        (GOOD.replace("https://huggingface.co/x/y", "http://example.com"), "source_url"),
        (GOOD.replace('"Apache-2.0"', '"other"'), "license_url"),
        (GOOD.replace('"Apache-2.0"', '"other"\nlicense_url = "http://x.org"'), "license_url"),
        (GOOD.replace('"Apache-2.0"', '""'), "license"),
        (GOOD.replace('reason = "Known tokenizer bug."', 'reason = "  "'), "reason"),
        (GOOD.replace('"hold"', '"keep"'), "verdict"),
        (GOOD.replace('run = "2026', 'run = "other-2026'), "run"),
    ],
)
def test_rejections(tmp_path, tiers, text, match):
    with pytest.raises(WeekInputError, match=match):
        load_week(write(tmp_path, text), tiers)


def test_other_license_with_url(tmp_path, tiers):
    text = GOOD.replace('"Apache-2.0"', '"other"\nlicense_url = "https://x.org/l"')
    assert load_week(write(tmp_path, text), tiers).runs[0].license_url == "https://x.org/l"


def tier_block(i, name="a", env="e"):
    return f'[[tier]]\nid = "{i}"\nname = "{name}"\nenvelope = "{env}"\n'


@pytest.mark.parametrize(
    "body,match",
    [
        (tier_block("a", env="e1") + tier_block("a", env="e2"), "duplicate tier id"),
        (tier_block("a") + tier_block("b"), "duplicate envelope"),
        (tier_block("Bad Id"), "slug"),
        ("", "tier"),
        (tier_block("a") + "oops = 1\n", "oops"),
    ],
)
def test_tier_rejections(tmp_path, body, match):
    with pytest.raises(WeekInputError, match=match):
        load_tiers(write(tmp_path, body, "tiers.toml"))


SITE = (ROOT / "site.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "text,match",
    [
        (SITE.replace("https://bakeoff.si", "http://bakeoff.si"), "base_url"),
        (SITE.replace("https://github.com", "ftp://github.com"), "repo_url"),
        (SITE.replace('action = ""', 'action = "http://x.org/f"'), "action"),
        (SITE.replace("launched = false", 'launched = "no"'), "launched"),
        (SITE + "extra = 1\n", "extra"),
        (SITE.replace("launched = false\n", ""), "launched"),
    ],
)
def test_site_rejections(tmp_path, text, match):
    with pytest.raises(WeekInputError, match=match):
        load_site(write(tmp_path, text, "site.toml"))


def test_site_https_action(tmp_path):
    text = SITE.replace('action = ""', 'action = "https://x.org/f"')
    assert load_site(write(tmp_path, text, "site.toml")).newsletter_action == "https://x.org/f"


@pytest.mark.parametrize(
    "url",
    [
        "https://u:tok@news.example.org/x",
        "https://a.example;script-src/x",
        "https:///path",
        "https://",
        "https://exa mple.org/",
        "https://example.org/\tx",
        "https://example.org:notaport/",
        "https://Ex_ample.org/",
    ],
)
def test_hostile_urls_rejected(tmp_path, tiers, url):
    for field, text in (
        ("base_url", SITE.replace("https://bakeoff.si", url)),
        ("newsletter.action", SITE.replace('action = ""', f'action = "{url}"')),
    ):
        with pytest.raises(WeekInputError, match=field.split(".")[-1]):
            load_site(write(tmp_path, text, "site.toml"))
    with pytest.raises(WeekInputError, match="source_url"):
        load_week(write(tmp_path, GOOD.replace("https://huggingface.co/x/y", url)), tiers)


def test_url_with_port_is_accepted(tmp_path):
    text = SITE.replace('action = ""', 'action = "https://news.example.org:8443/f"')  # scrub-ok
    assert load_site(write(tmp_path, text, "site.toml")).newsletter_action.endswith("/f")


def test_override_tier_must_match_its_runs_tier(tmp_path, tiers):
    text = GOOD.replace('tier = "24gb"\ntrack', 'tier = "12gb"\ntrack')
    with pytest.raises(WeekInputError, match=r"override\[0\]\.tier '12gb'.*run .*'24gb'"):
        load_week(write(tmp_path, text), tiers)
