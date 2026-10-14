import json
from datetime import UTC, datetime

import pytest

from bakeoff.bundle import run_id, scan_text, slug, verify_bundle, write_bundle


def _kinds(findings):
    return {f.kind for f in findings}


def test_slug():
    assert slug("Example-14B AWQ!") == "example-14b-awq"
    assert len(slug("a" * 100)) == 40
    assert not slug("--x--").startswith("-")


def test_run_id_format():
    rid = run_id(datetime(2026, 10, 13, 15, 30, tzinfo=UTC), "Example-14B AWQ!", b"x")
    assert rid.startswith("20261013T153000Z-example-14b-awq-") and len(rid.split("-")[-1]) == 8


def test_round_trip_verifies_with_unicode(tmp_path):
    write_bundle(
        tmp_path,
        {"status": "complete", "bundle_format": "m1"},
        {"quality/general.x.jsonl": [{"output": "naïve 🍰"}]},
    )
    assert verify_bundle(tmp_path, username=None, denylist=None) == []


def test_tampered_file_detected(tmp_path):
    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    (tmp_path / "speed.jsonl").write_text('{"a": 2}\n', encoding="utf-8")
    assert "hash_mismatch" in _kinds(verify_bundle(tmp_path, None, None))


def test_tampered_meta_detected(tmp_path):
    write_bundle(tmp_path, {"status": "complete", "x": 1}, {"speed.jsonl": [{"a": 1}]})
    p = tmp_path / "bundle.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["x"] = 2
    p.write_text(json.dumps(meta), encoding="utf-8")
    assert "bundle_hash" in _kinds(verify_bundle(tmp_path, None, None))


def test_missing_and_unlisted_files(tmp_path):
    write_bundle(
        tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}], "fit.jsonl": [{"b": 1}]}
    )
    (tmp_path / "fit.jsonl").unlink()
    (tmp_path / "extra.jsonl").write_text("{}\n", encoding="utf-8")
    assert {"missing_file", "unlisted_file"} <= _kinds(verify_bundle(tmp_path, None, None))


def test_incomplete_flagged(tmp_path):
    write_bundle(tmp_path, {"status": "incomplete"}, {"speed.jsonl": [{"a": 1}]})
    assert "incomplete" in _kinds(verify_bundle(tmp_path, None, None))


def test_verify_scans_content_and_denylist(tmp_path):
    write_bundle(
        tmp_path,
        {"status": "complete"},
        {"speed.jsonl": [{"a": "see 10.1.2.3 and alice and lab-box-7"}]},  # scrub-ok
    )
    deny = tmp_path.parent / (tmp_path.name + "-deny.txt")
    deny.write_text("# comment\n\nlab-box-7\n", encoding="utf-8")  # scrub-ok
    kinds = _kinds(verify_bundle(tmp_path, username="alice", denylist=deny))  # scrub-ok
    assert {"private_ip", "username", "denylist"} <= kinds
    missing = tmp_path.parent / "nope.txt"
    assert "denylist" not in _kinds(verify_bundle(tmp_path, None, missing))


@pytest.mark.parametrize(
    "text,kind",
    [
        ("ip 10.1.2.3", "private_ip"),  # scrub-ok
        ("ip 192.168.0.9", "private_ip"),  # scrub-ok
        ("ip 100.64.0.1", "private_ip"),  # scrub-ok
        ("ip 172.20.0.1", "private_ip"),  # scrub-ok
        ("ip 169.254.1.1", "private_ip"),  # scrub-ok
        ("ip 127.0.0.1", "private_ip"),  # scrub-ok
        ("ip fd12:3456::1", "private_ip"),  # scrub-ok
        ("ip fe80::1", "private_ip"),  # scrub-ok
        ("http://localhost:8000/v1", "host_port"),  # scrub-ok
        ("lab-box-7.lan:11434", "host_port"),  # scrub-ok
        ("/Users/alice/x", "home_path"),  # scrub-ok
        ("/home/alice", "home_path"),  # scrub-ok
        ("C:\\Users\\alice\\x", "home_path"),  # scrub-ok
    ],
)
def test_scan_text_flags(text, kind):
    assert kind in {k for _, k, _ in scan_text(text, [])}


@pytest.mark.parametrize(
    "text",
    [
        "ip 192.0.2.10",
        "ip 172.32.0.1",
        "ip 100.128.0.1",
        "gpt-oss:20b",
        "at 12:30:00",
        "Record 00012: ok",
        "Answer:72",
        "ratio:12",
        "the /Users/ folder",
    ],
)
def test_scan_text_clean(text):
    assert scan_text(text, []) == []


def test_line_numbers():
    hits = scan_text("ok\nip 10.0.0.1\n", [])  # scrub-ok
    assert [h[0] for h in hits] == [2]


def test_username_and_denylist_whole_word():
    hits = scan_text(
        "alice ran on lab-box-7; malice is fine",  # scrub-ok
        ["lab-box-7", "ab"],  # scrub-ok
        username="alice",  # scrub-ok
    )
    assert sorted(k for _, k, _ in hits) == ["denylist", "username"]
