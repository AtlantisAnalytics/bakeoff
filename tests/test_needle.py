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
