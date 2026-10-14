import json
import re
from pathlib import Path

from bakeoff import __version__
from bakeoff.bundle import scan_text
from bakeoff.config import (
    EndpointConfig,
    FitConfig,
    ModelConfig,
    QualityConfig,
    RunConfig,
    SpeedConfig,
)
from bakeoff.env import (
    capture_env,
    detect_platform,
    field,
    parse_cuda_version,
    parse_nvidia_query,
    parse_vm_stat,
)

DATA = Path(__file__).resolve().parent / "data" / "env"
LEAF = {"value", "source"}
SOURCES = {"measured", "reported", "operator", "not_captured"}


def fx(name):
    return (DATA / name).read_text()


CFG = RunConfig(
    endpoint=EndpointConfig(base_url="http://192.0.2.10:8000/v1", model="m"),  # scrub-ok
    model=ModelConfig(
        label="Example",
        weights="hf:example-org/Example@abc",
        weights_file="ex.gguf",
        quant="Q4_K_M",
        runtime="ollama",
        envelope="8k",
    ),
    quality=QualityConfig(),
    speed=SpeedConfig(),
    fit=FitConfig(),
)


def fake_run(table):
    """Runner keyed on argv[0] (and argv[1] for sysctl/git); records calls."""
    calls = []

    def run(argv):
        calls.append(list(argv))
        for key, out in table.items():
            if list(argv[: len(key)]) == list(key):
                return out
        return None

    run.calls = calls
    return run


def fake_read(files):
    return lambda path: files.get(path)


LINUX_FILES = {
    "/etc/os-release": fx("os-release.txt"),
    "/proc/cpuinfo": fx("cpuinfo.txt"),
    "/proc/meminfo": fx("meminfo.txt"),
}


def leaves(node, path=""):
    """Yield (path, leaf) for every {"value","source"} leaf; fail on any other shape."""
    if isinstance(node, dict) and set(node) == LEAF:
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, f"{path}[{i}]")
    else:
        raise AssertionError(f"non-leaf scalar at {path}: {node!r}")


def check_shape(env):
    assert set(env) == {"host", "gpus", "runtime", "model", "harness"}
    for path, leaf in leaves(env):
        assert leaf["source"] in SOURCES, path
        if leaf["value"] is None:
            assert leaf["source"] == "not_captured", path


def nvidia_run(query, git=True):
    run_q = fake_run({})

    def run(argv):
        run_q.calls.append(list(argv))
        if argv == ["nvidia-smi", "-L"]:
            return "GPU 0: redacted\n"
        if argv == ["nvidia-smi"]:
            return fx("nvidia_smi_header.txt")
        if argv[0] == "nvidia-smi" and argv[1].startswith("--query-gpu="):
            return query
        if argv[:1] == ["git"]:
            if not git:
                return None
            if "rev-parse" in argv:
                return "0123456789abcdef0123456789abcdef01234567\n"
            if "status" in argv:
                return " M src/x.py\n"
        return None

    run.calls = run_q.calls
    return run


def test_field_none_is_not_captured():
    assert field(None, "measured") == {"value": None, "source": "not_captured"}
    assert field(0, "measured") == {"value": 0, "source": "measured"}


def test_nvidia_host():
    run = nvidia_run(fx("nvidia_query_1gpu.csv"))
    env = capture_env(
        CFG, {"version": "0.9.1"}, "served", run=run, read=fake_read(LINUX_FILES), platform="linux"
    )
    check_shape(env)
    gpu = env["gpus"][0]
    assert gpu["name"] == {"value": "NVIDIA GeForce RTX 3090 Ti", "source": "measured"}
    assert gpu["vram_mib"]["value"] == 24564
    assert gpu["driver"]["value"] == "570.86.15"
    assert gpu["cuda"] == {"value": "12.8", "source": "measured"}
    assert gpu["power_limit_w"] == {"value": 450.0, "source": "measured"}
    assert set(gpu) == {"name", "vram_mib", "driver", "cuda", "power_limit_w"}
    assert env["host"]["os"]["value"] == "Linux"
    assert env["host"]["os_version"]["value"] == "Ubuntu 24.04.1 LTS"
    assert env["host"]["cpu"]["value"] == "AMD Ryzen 9 7950X 16-Core Processor"
    assert env["host"]["cpu_cores"]["value"] == 2
    assert env["host"]["ram_bytes"]["value"] == 65800000 * 1024
    assert env["runtime"]["version"] == {"value": "0.9.1", "source": "reported"}
    assert env["runtime"]["served_model"] == {"value": "served", "source": "reported"}
    h = env["harness"]
    assert h["version"] == {"value": __version__, "source": "measured"}
    assert h["git_commit"]["value"] == "0123456789abcdef0123456789abcdef01234567"
    assert h["git_dirty"] == {"value": True, "source": "measured"}
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", h["python"]["value"])
    # momentary sampler values are never recorded
    assert "1021" not in json.dumps(env) and "38.52" not in json.dumps(env)


def test_nvidia_three_gpus_with_na_power():
    run = nvidia_run(fx("nvidia_query_3gpu.csv"))
    env = capture_env(CFG, None, None, run=run, read=fake_read(LINUX_FILES), platform="linux")
    check_shape(env)
    assert [g["name"]["value"] for g in env["gpus"]] == [
        "NVIDIA GeForce RTX 3090 Ti",
        "NVIDIA GeForce RTX 3060",
        "NVIDIA GeForce RTX 3060",
    ]
    assert [g["power_limit_w"]["value"] for g in env["gpus"]] == [450.0, 170.0, 170.0]
    rows = parse_nvidia_query(fx("nvidia_query_3gpu.csv"))
    assert rows[1]["power_w"] is None and rows[2]["power_w"] == 14.10
    assert rows[0] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3090 Ti",
        "vram_mib": 24564,
        "driver": "570.86.15",
        "power_limit_w": 450.0,
        "mem_used_mib": 1021,
        "power_w": 38.52,
    }


def test_na_power_limit_is_not_captured():
    csv = "0, NVIDIA GeForce RTX 3060, 12288, 570.86.15, [N/A], 512, [N/A]\n"
    env = capture_env(
        CFG, None, None, run=nvidia_run(csv), read=fake_read(LINUX_FILES), platform="linux"
    )
    check_shape(env)
    assert env["gpus"][0]["power_limit_w"] == {"value": None, "source": "not_captured"}


def test_darwin_host():
    run = fake_run(
        {
            ("sw_vers", "-productVersion"): fx("sw_vers.txt"),
            ("sysctl", "-n"): fx("sysctl_darwin.txt"),
        }
    )
    env = capture_env(CFG, None, "m", run=run, read=fake_read({}), platform="darwin")
    check_shape(env)
    assert env["gpus"] == [
        {
            "name": {"value": "Apple M4 Max", "source": "measured"},
            "unified_memory_bytes": {"value": 38654705664, "source": "measured"},
        }
    ]
    assert env["host"]["os"]["value"] == "macOS"
    assert env["host"]["os_version"]["value"] == "15.6.1"
    assert env["host"]["cpu"]["value"] == "Apple M4 Max"
    assert env["host"]["cpu_cores"]["value"] == 14
    assert env["host"]["ram_bytes"]["value"] == 38654705664
    assert detect_platform(run, "darwin") == "darwin"


def test_nvidia_smi_broken():
    # not installed, or exiting non-zero after a driver mismatch: the runner yields None
    run = fake_run({})
    assert detect_platform(run, "linux") == "none"
    env = capture_env(CFG, None, None, run=run, read=fake_read(LINUX_FILES), platform="linux")
    check_shape(env)
    assert env["gpus"] == []
    assert env["host"]["cpu"]["value"] == "AMD Ryzen 9 7950X 16-Core Processor"


def test_detect_platform_nvidia_wins():
    assert detect_platform(fake_run({("nvidia-smi", "-L"): "GPU 0: x\n"}), "linux") == "nvidia"


def test_query_failure_after_detect_gives_no_gpus():
    run = fake_run({("nvidia-smi", "-L"): "GPU 0: x\n"})
    env = capture_env(CFG, None, None, run=run, read=fake_read(LINUX_FILES), platform="linux")
    check_shape(env)
    assert env["gpus"] == []


def test_operator_fields():
    env = capture_env(CFG, None, None, run=fake_run({}), read=fake_read({}), platform="linux")
    check_shape(env)
    for k, v in {
        "label": "Example",
        "weights": "hf:example-org/Example@abc",
        "weights_file": "ex.gguf",
        "quant": "Q4_K_M",
        "envelope": "8k",
    }.items():
        assert env["model"][k] == {"value": v, "source": "operator"}
    assert env["runtime"]["name"] == {"value": "ollama", "source": "operator"}
    assert env["runtime"]["version"] == {"value": None, "source": "not_captured"}
    assert env["runtime"]["served_model"]["source"] == "not_captured"
    assert env["host"]["os_version"]["source"] == "not_captured"


def test_no_git_gives_not_captured():
    run = nvidia_run(fx("nvidia_query_1gpu.csv"), git=False)
    env = capture_env(CFG, None, None, run=run, read=fake_read(LINUX_FILES), platform="linux")
    assert env["harness"]["git_commit"]["source"] == "not_captured"
    assert env["harness"]["git_dirty"]["source"] == "not_captured"


def test_clean_tree():
    def run(argv):
        if argv[:1] == ["git"]:
            return "0123456789abcdef0123456789abcdef01234567\n" if "rev-parse" in argv else ""
        return None

    env = capture_env(CFG, None, None, run=run, read=fake_read({}), platform="linux")
    assert env["harness"]["git_dirty"] == {"value": False, "source": "measured"}


def test_git_path_not_recorded():
    run = nvidia_run(fx("nvidia_query_1gpu.csv"))
    env = capture_env(CFG, None, None, run=run, read=fake_read(LINUX_FILES), platform="linux")
    git_calls = [c for c in run.calls if c[0] == "git"]
    assert git_calls and all(c[1] == "-C" for c in git_calls)
    assert git_calls[0][2] not in json.dumps(env)


def test_only_allowlisted_commands_run():
    run = nvidia_run(fx("nvidia_query_1gpu.csv"))
    capture_env(CFG, None, None, run=run, read=fake_read(LINUX_FILES), platform="linux")
    for argv in run.calls:
        assert argv[0] in {"nvidia-smi", "git"}
        if argv[0] == "nvidia-smi":
            assert len(argv) == 1 or argv[1] == "-L" or argv[1].startswith("--query-gpu=")
    mac = fake_run({("sw_vers",): "15.6.1\n", ("sysctl",): fx("sysctl_darwin.txt")})
    capture_env(CFG, None, None, run=mac, read=fake_read({}), platform="darwin")
    assert {c[0] for c in mac.calls} <= {"nvidia-smi", "sw_vers", "sysctl", "git"}


def test_never_leaks_identity():
    def leaky_run(argv):
        if argv[0] in ("hostname", "uname"):
            return "lab-box-7\n"  # scrub-ok
        if argv[0] == "nvidia-smi" and argv[1:2] == ["-L"]:
            return "GPU 0: RTX 3090 Ti (UUID: GPU-0a1b2c3d-0000-1111-2222-333344445555)\n"
        if argv[0] == "nvidia-smi" and len(argv) == 1:
            return fx("nvidia_smi_header.txt") + "host lab-box-7 /home/alice/x\n"  # scrub-ok
        if argv[0] == "nvidia-smi":
            return fx("nvidia_query_1gpu.csv")
        if argv[0] == "git":
            return "lab-box-7 /home/alice/bakeoff\n"  # scrub-ok
        return None

    def leaky_read(path):
        text = LINUX_FILES.get(path, "")
        return text + "\nhostname: lab-box-7\nuser alice /home/alice\n"  # scrub-ok

    env = capture_env(CFG, None, "m", run=leaky_run, read=leaky_read, platform="linux")
    text = json.dumps(env)
    assert "lab-box-7" not in text and "alice" not in text  # scrub-ok
    assert "UUID" not in text and "GPU-0a1b" not in text
    assert scan_text(text, ["lab-box-7", "alice"]) == []  # scrub-ok
    check_shape(env)


def test_parse_cuda_version():
    assert parse_cuda_version(fx("nvidia_smi_header.txt")) == "12.8"
    assert parse_cuda_version("nothing here") is None


def test_parse_vm_stat():
    assert parse_vm_stat(fx("vm_stat.txt")) == (90000 + 200000 + 30000) * 16384
    assert parse_vm_stat("garbage") is None
