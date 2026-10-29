import re

from bakeoff.needle import build_haystack, needle_messages


def test_haystack_deterministic():
    assert build_haystack(50, 3) == build_haystack(50, 3) != build_haystack(50, 4)
    assert re.fullmatch(
        r"Record 00000: the .+ depot logged \d+ crates of .+ on day \d+\.", build_haystack(1, 1)[0]
    )


def test_needle_position_and_key():
    msgs, key = needle_messages(100, 0.5, 7)
    lines = msgs[0]["content"].split("\n")
    assert f"The access code for the vault is {key}." == lines[50]
    assert re.fullmatch(r"[A-Z]+-\d{4}", key)
    assert msgs[0]["content"].endswith("Reply with the code only.")
    assert needle_messages(100, 0.5, 7) == (msgs, key)


def test_needle_depth_edges():
    msgs, key = needle_messages(10, 0.0, 1)
    assert msgs[0]["content"].startswith(f"The access code for the vault is {key}.")
    msgs, key = needle_messages(10, 1.0, 1)
    assert msgs[0]["content"].split("\n")[10] == f"The access code for the vault is {key}."


def test_golden_values():
    assert build_haystack(5, 3)[0] == (
        "Record 00000: the Hollowmere depot logged 608 crates of wool on day 67."
    )
    assert needle_messages(100, 0.5, 7)[1] == "BEACON-4071"


# --- vaults: the long-context suite from version 2 -------------------------------------------

from bakeoff.graders import grade  # noqa: E402
from bakeoff.needle import VAULT_KINDS, vault_messages  # noqa: E402

STATEMENT = re.compile(r"The access code for the (.+?) vault is ([A-Z]+-\d{4})\.")


def _vault_lines(msgs):
    return [ln for ln in msgs[0]["content"].split("\n") if STATEMENT.fullmatch(ln)]


def test_vaults_deterministic():
    assert vault_messages(200, 0.5, "changed", 7) == vault_messages(200, 0.5, "changed", 7)
    assert vault_messages(200, 0.5, "changed", 7) != vault_messages(200, 0.5, "changed", 8)


def test_vaults_eight_vaults_with_distinct_codes():
    for kind in VAULT_KINDS:
        msgs, _ = vault_messages(200, 0.5, kind, 7)
        found = [STATEMENT.fullmatch(ln).groups() for ln in _vault_lines(msgs)]
        assert len(found) == 8
        assert len({name for name, _ in found}) == 8 and len({c for _, c in found}) == 8


def test_vaults_distractors_kind():
    msgs, expected = vault_messages(200, 0.5, "distractors", 7)
    lines = msgs[0]["content"].split("\n")
    town = re.search(r"current access code for the (.+?) vault\?", lines[-1]).group(1)
    codes = dict(STATEMENT.fullmatch(ln).groups() for ln in _vault_lines(msgs))
    assert expected == {
        "answerable": True,
        "must_include": [codes[town]],
        "must_not_include": sorted(c for n, c in codes.items() if n != town),
    }
    assert f"{town} annex" in codes  # a near-name distractor
    at = next(i for i, ln in enumerate(lines) if ln.startswith(f"The access code for the {town} v"))
    assert abs(at / (len(lines) - 1) - 0.5) < 0.05


def test_vaults_changed_kind_asks_for_the_new_code():
    msgs, expected = vault_messages(200, 0.9, "changed", 7)
    text = msgs[0]["content"]
    town = re.search(r"current access code for the (.+?) vault\?", text).group(1)
    old = dict(STATEMENT.fullmatch(ln).groups() for ln in _vault_lines(msgs))[town]
    new = re.search(
        rf"the access code for the {town} vault was changed to ([A-Z]+-\d{{4}})\.", text
    ).group(1)
    assert text.index(old) < text.index(new) and old != new
    assert expected["answerable"] and expected["must_include"] == [new]
    assert old in expected["must_not_include"] and len(expected["must_not_include"]) == 8


def test_vaults_absent_kind_only_has_the_annex():
    msgs, expected = vault_messages(200, 0.1, "absent", 7)
    town = re.search(r"current access code for the (.+?) vault\?", msgs[0]["content"]).group(1)
    names = [STATEMENT.fullmatch(ln).group(1) for ln in _vault_lines(msgs)]
    assert town not in names and f"{town} annex" in names
    assert expected == {"answerable": False, "must_include": [], "must_not_include": []}


def test_vaults_grade_with_the_grounding_grader():
    for kind in VAULT_KINDS:
        msgs, e = vault_messages(300, 0.5, kind, 3)
        right = e["must_include"][0] if e["answerable"] else "NOT IN CONTEXT"
        assert grade("grounding", right, e, {}).passed, kind
        every_code = " ".join(STATEMENT.fullmatch(ln).group(2) for ln in _vault_lines(msgs))
        assert not grade("grounding", every_code, e, {}).passed, kind
