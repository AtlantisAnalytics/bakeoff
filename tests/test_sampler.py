import json
import time

from bakeoff.sampler import Sampler

GOOD_VM = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               1000.
Pages active:                             {active}.
Pages wired down:                         {wired}.
Pages occupied by compressor:             {comp}.
"""


def feed(*outs):
    it = iter(outs)
    return lambda cmd: next(it)


def test_peaks_by_pass():
    s = Sampler(
        "nvidia",
        feed("0, 1000, 100\n1, 500, 50", "0, 3000, 300\n1, 700, 80", "0, 2000, 200\n1, 600, 60"),
    )
    s.sample_once()
    s.mark("speed")
    s.sample_once()
    s.sample_once()
    p = s.peaks()
    assert p["baseline"]["mem_used_mib_total"] == 1500
    assert p["speed"]["mem_used_mib_total"] == 3700
    assert p["speed"]["mem_used_mib_per_gpu"] == [3000, 700]
    assert p["speed"]["power_w_max"] == 380
    assert p["speed"]["power_w_mean"] == (380 + 260) / 2
    assert s.captured


def test_nvidia_command_and_record_shape():
    seen = []

    def run(cmd):
        seen.append(cmd)
        return "0, 1000, 100.5"

    rec = Sampler("nvidia", run).sample_once()
    assert seen == [
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
    ]
    assert rec["pass"] == "baseline"
    assert rec["gpus"] == [{"index": 0, "mem_used_mib": 1000, "power_w": 100.5}]
    assert rec["t"] >= 0
    json.dumps(rec)


def test_power_na():
    s = Sampler("nvidia", feed("0, 1000, [N/A]\n1, 500, 50"))
    rec = s.sample_once()
    assert rec["gpus"][0]["power_w"] is None
    p = s.peaks()["baseline"]
    assert p["mem_used_mib_total"] == 1500
    assert p["power_w_mean"] is None and p["power_w_max"] is None


def test_power_only_over_fully_numeric_samples():
    s = Sampler(
        "nvidia", feed("0, 1, 100\n1, 1, 50", "0, 1, [N/A]\n1, 1, 900", "0, 1, 10\n1, 1, 20")
    )
    for _ in range(3):
        s.sample_once()
    p = s.peaks()["baseline"]
    assert p["power_w_max"] == 150
    assert p["power_w_mean"] == 90


def test_none_platform_records_nothing():
    calls = []
    s = Sampler("none", lambda c: calls.append(c) or "x")
    assert s.sample_once() is None
    s.start()
    assert s.stop() == []
    assert not s.captured and s.peaks() == {}
    assert calls == []


def test_failing_command_skipped():
    s = Sampler("nvidia", lambda cmd: None)
    assert s.sample_once() is None
    assert not s.captured and s.peaks() == {}


def test_run_raising_is_swallowed():
    def boom(cmd):
        raise RuntimeError("driver mismatch")

    s = Sampler("nvidia", boom, interval_s=0.01)
    assert s.sample_once() is None
    s.start()
    time.sleep(0.05)
    assert s.stop() == []


def test_thread_survives_intermittent_exceptions():
    n = {"i": 0}

    def flaky(cmd):
        n["i"] += 1
        if n["i"] % 2:
            raise OSError("flaky")
        return "0, 100, 10"

    s = Sampler("nvidia", flaky, interval_s=0.01)
    s.start()
    time.sleep(0.15)
    assert len(s.stop()) >= 2


def test_malformed_rows_skip_whole_sample():
    bad = [
        "0, 1000",  # wrong field count
        "0, 1000, 50, 7",
        "0, abc, 50",  # non-numeric memory
        "0, 1000, 50\n1, 5",  # one bad row spoils the sample
        "",
        "garbage",
        "x, 1000, 50",  # non-numeric index
    ]
    s = Sampler("nvidia", feed(*bad))
    for _ in bad:
        assert s.sample_once() is None
    assert not s.captured


def test_non_numeric_or_nonfinite_power_is_none():
    s = Sampler("nvidia", feed("0, 100, nan", "0, 100, abc"))
    assert s.sample_once()["gpus"][0]["power_w"] is None
    assert s.sample_once()["gpus"][0]["power_w"] is None


def test_darwin_peaks():
    s = Sampler(
        "darwin",
        feed(
            GOOD_VM.format(active=10, wired=20, comp=30),
            GOOD_VM.format(active=100, wired=20, comp=30),
            "not vm_stat output",
        ),
    )
    r = s.sample_once()
    assert r["mem_used_bytes"] == 60 * 16384 and r["pass"] == "baseline"
    s.mark("quality")
    s.sample_once()
    assert s.sample_once() is None
    p = s.peaks()
    assert p["baseline"] == {"mem_used_bytes": 60 * 16384}
    assert p["quality"] == {"mem_used_bytes": 150 * 16384}


def test_darwin_command():
    seen = []
    Sampler("darwin", lambda c: seen.append(c)).sample_once()
    assert seen == [["vm_stat"]]


def test_timestamps_monotonic_since_start():
    s = Sampler("nvidia", lambda c: "0, 1, 1")
    s.start()
    time.sleep(0.02)
    s.sample_once()
    recs = s.stop()
    assert recs[0]["t"] < 0.02 <= recs[-1]["t"]
    assert all(b["t"] >= a["t"] for a, b in zip(recs, recs[1:], strict=False))


def test_thread_start_stop():
    s = Sampler("nvidia", lambda c: "0, 1000, 100", interval_s=0.01)
    s.start()
    time.sleep(0.05)
    recs = s.stop()
    assert len(recs) >= 3
    assert s.stop() == recs  # idempotent
    assert s._thread is not None and not s._thread.is_alive()
    n = len(recs)
    time.sleep(0.03)
    assert len(s.stop()) == n
    json.dumps(recs)


def test_stop_returns_copy_and_start_idempotent():
    s = Sampler("nvidia", lambda c: "0, 1, 1", interval_s=0.01)
    s.start()
    s.start()
    recs = s.stop()
    recs.clear()
    assert s.stop() != []


def test_stop_without_start():
    s = Sampler("nvidia", lambda c: "0, 1, 1")
    s.sample_once()
    assert len(s.stop()) == 1


def test_peaks_concurrent_with_thread():
    s = Sampler("nvidia", lambda c: "0, 1000, 100", interval_s=0.001)
    s.start()
    for _ in range(50):
        s.peaks()
    s.stop()


def test_ragged_gpu_counts():
    s = Sampler("nvidia", feed("0, 10, 1", "0, 5, 1\n1, 20, 1"))
    s.sample_once()
    s.sample_once()
    p = s.peaks()["baseline"]
    assert p["mem_used_mib_per_gpu"] == [10, 20]
    assert p["mem_used_mib_total"] == 25
