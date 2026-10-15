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
        {"status": "complete", "bundle_format": "m2"},
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
    run = tmp_path / "run"
    write_bundle(
        run,
        {"status": "complete"},
        {"speed.jsonl": [{"a": "see 10.1.2.3 and alice and lab-box-7"}]},  # scrub-ok
    )
    deny_dir = tmp_path / "cfg"
    deny_dir.mkdir()
    deny = deny_dir / "denylist.txt"
    deny.write_text("# comment\n\nlab-box-7\n", encoding="utf-8")  # scrub-ok
    kinds = _kinds(verify_bundle(run, username="alice", denylist=deny))  # scrub-ok
    assert {"private_ip", "username", "denylist"} <= kinds
    # the same content has no denylist finding when the denylist path is absent
    assert "denylist" not in _kinds(verify_bundle(run, None, deny_dir / "nope.txt"))


def test_json_escaped_windows_path_flagged(tmp_path):
    write_bundle(
        tmp_path,
        {"status": "complete"},
        {"speed.jsonl": [{"o": "C:\\Users\\alice\\x"}]},  # scrub-ok
    )
    assert "home_path" in _kinds(verify_bundle(tmp_path, None, None))


def test_decoded_denylist_and_unicode_escape(tmp_path):
    write_bundle(
        tmp_path,
        {"status": "complete"},
        {"speed.jsonl": [{"o": 'say "my\\host" é'}]},
    )
    deny = tmp_path / "d.txt"
    deny.write_text("my\\host\n", encoding="utf-8")  # scrub-ok
    assert "denylist" in _kinds(verify_bundle(tmp_path, None, deny))


def test_non_json_files_scanned(tmp_path):
    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    (tmp_path / "notes.TXT").write_text("ip 10.9.8.7\n", encoding="utf-8")  # scrub-ok
    kinds = _kinds(verify_bundle(tmp_path, None, None))
    assert {"unlisted_file", "private_ip"} <= kinds


def test_corrupt_bundle_json(tmp_path):
    (tmp_path / "bundle.json").write_text("{nope", encoding="utf-8")
    assert _kinds(verify_bundle(tmp_path, None, None)) == {"unreadable"}


def test_non_dict_meta(tmp_path):
    (tmp_path / "bundle.json").write_text("[1]", encoding="utf-8")
    assert _kinds(verify_bundle(tmp_path, None, None)) == {"unreadable"}


def test_non_dict_files(tmp_path):
    (tmp_path / "bundle.json").write_text('{"status": "complete", "files": []}', encoding="utf-8")
    assert "unreadable" in _kinds(verify_bundle(tmp_path, None, None))


def test_non_utf8_file(tmp_path):
    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")
    assert "unreadable" in _kinds(verify_bundle(tmp_path, None, None))


def test_malformed_jsonl_line_unreadable(tmp_path):
    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    (tmp_path / "speed.jsonl").write_text('{"a": 1}\n{oops\n', encoding="utf-8")
    f = [x for x in verify_bundle(tmp_path, None, None) if x.kind == "unreadable"]
    assert [(x.file, x.line) for x in f] == [("speed.jsonl", 2)]


def test_symlinked_directory_in_key(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "x.jsonl").write_text("{}\n", encoding="utf-8")
    run = tmp_path / "run"
    write_bundle(run, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    (run / "linkdir").symlink_to(real, target_is_directory=True)
    p = run / "bundle.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["files"]["linkdir/x.jsonl"] = "0" * 64
    p.write_text(json.dumps(meta), encoding="utf-8")
    findings = verify_bundle(run, None, None)
    assert "linkdir/x.jsonl" in {f.file for f in findings if f.kind == "unreadable"}
    assert "hash_mismatch" not in _kinds(findings)


def test_deeply_nested_json_unreadable(tmp_path):
    deep = "[" * 100000 + "]" * 100000
    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    (tmp_path / "speed.jsonl").write_text(deep + "\n", encoding="utf-8")
    (tmp_path / "extra.json").write_text(deep, encoding="utf-8")
    kinds = _kinds(verify_bundle(tmp_path, None, None))
    assert "unreadable" in kinds
    (tmp_path / "bundle.json").write_text(deep, encoding="utf-8")
    assert _kinds(verify_bundle(tmp_path, None, None)) == {"unreadable"}


def test_unreadable_subdirectory(tmp_path):
    import os

    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    sub = tmp_path / "locked"
    sub.mkdir()
    sub.chmod(0)
    try:
        if os.access(sub, os.R_OK):
            pytest.skip("cannot make directory unreadable")
        assert "unreadable" in _kinds(verify_bundle(tmp_path, None, None))
    finally:
        sub.chmod(0o700)


def test_write_bundle_rejects_unsafe_paths(tmp_path):
    for bad in ("../x.jsonl", "/abs/x.jsonl", "a/../../x.jsonl"):
        with pytest.raises(ValueError):
            write_bundle(tmp_path / "r", {"status": "complete"}, {bad: [{}]})


def test_verify_unsafe_key_and_symlink(tmp_path):
    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    p = tmp_path / "bundle.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["files"]["../evil.jsonl"] = "0" * 64
    p.write_text(json.dumps(meta), encoding="utf-8")
    outside = tmp_path.parent / (tmp_path.name + "-out.txt")
    outside.write_text("x", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    findings = verify_bundle(tmp_path, None, None)
    unreadable = {f.file for f in findings if f.kind == "unreadable"}
    assert {"../evil.jsonl", "link.txt"} <= unreadable


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
        ("c:\\users\\alice", "home_path"),  # scrub-ok
        ('path="/home/alice/x"', "home_path"),  # scrub-ok
        ("a=/Users/alice", "home_path"),  # scrub-ok
        ("file:///Users/alice/x", "home_path"),  # scrub-ok
        ("file:///home/alice", "home_path"),  # scrub-ok
        ("alice_laptop", "username"),  # scrub-ok
    ],
)
def test_scan_text_flags(text, kind):
    assert kind in {k for _, k, _ in scan_text(text, [], username="alice")}  # scrub-ok


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
        "https://example.com/home/page",
        "see example.com/Users/page",
    ],
)
def test_scan_text_clean(text):
    assert scan_text(text, []) == []


def test_line_numbers_split_on_newline_only():
    hits = scan_text("a\x0cb\u2028c\nip 10.0.0.1", [])  # scrub-ok
    assert [h[0] for h in hits] == [2]


def test_underscore_is_separator():
    hits = scan_text("lab-box-7_gpu", ["lab-box-7"])  # scrub-ok
    assert [k for _, k, _ in hits] == ["denylist"]


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


def test_non_finite_numbers_are_refused(tmp_path):
    from bakeoff.hashing import canonical_json

    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        write_bundle(tmp_path / "a", {"status": "complete"}, {"s.jsonl": [{"x": float("inf")}]})
    with pytest.raises(ValueError):
        write_bundle(tmp_path / "b", {"status": "complete", "x": float("nan")}, {})


def test_verify_survives_nan_in_a_hand_edited_bundle(tmp_path):
    write_bundle(tmp_path, {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    p = tmp_path / "bundle.json"
    p.write_text(p.read_text().replace('"complete"', '"complete", "x": NaN'))
    assert "bundle_hash" in _kinds(verify_bundle(tmp_path, None, None))


def test_write_bundle_refuses_a_non_empty_run_dir(tmp_path):
    write_bundle(tmp_path / "r", {"status": "complete"}, {"speed.jsonl": [{"a": 1}]})
    with pytest.raises(FileExistsError):
        write_bundle(tmp_path / "r", {"status": "complete"}, {"speed.jsonl": [{"a": 2}]})
    (tmp_path / "empty").mkdir()
    write_bundle(tmp_path / "empty", {"status": "complete"}, {})  # an empty directory is fine


def test_run_id_includes_the_pass_selection():
    t = datetime(2026, 10, 13, 15, 30, tzinfo=UTC)
    assert "-demo-speed-" in run_id(t, "Demo", b"x", "speed")
    assert "-demo-" in run_id(t, "Demo", b"x") and "speed" not in run_id(t, "Demo", b"x")


def test_username_skip_scopes_to_output(tmp_path):
    write_bundle(
        tmp_path,
        {"status": "complete", "config": {"model": {"label": "alice-tuned"}}},  # scrub-ok
        {"quality/x.y.jsonl": [{"output": "alice went home"}]},  # scrub-ok
    )
    # a file is reported once by the raw-text pass and once by the per-string pass; count files
    skip = verify_bundle(tmp_path, "alice", None, username_skip_output=True)
    assert {f.file for f in skip if f.kind == "username"} == {"bundle.json"}
    full = verify_bundle(tmp_path, "alice", None, username_skip_output=False)
    assert {f.file for f in full if f.kind == "username"} == {"bundle.json", "quality/x.y.jsonl"}


def test_username_skip_keeps_other_patterns_in_output(tmp_path):
    deny = tmp_path / "deny.txt"
    deny.write_text("lab-box-7\n", encoding="utf-8")  # scrub-ok
    run = tmp_path / "run"
    write_bundle(
        run,
        {"status": "complete"},
        {
            "quality/x.y.jsonl": [
                {"output": "see /Users/x/notes and alice"},  # scrub-ok
                {"output": "host lab-box-7 here"},  # scrub-ok
            ]
        },
    )
    kinds = _kinds(verify_bundle(run, "alice", deny, username_skip_output=True))  # scrub-ok
    assert "home_path" in kinds and "denylist" in kinds
    assert "username" not in kinds
