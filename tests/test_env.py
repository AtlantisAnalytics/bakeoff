import json
import re
import sys
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


def make_repo(path, name="bakeoff"):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
    return path


def git_answers(toplevel, dirty=True):
    def git(argv):
        if "--show-toplevel" in argv:
            return f"{toplevel}\n"
        if "rev-parse" in argv:
            return "0123456789abcdef0123456789abcdef01234567\n"
        if "status" in argv:
            return " M src/x.py\n" if dirty else ""
        return None

    return git


def nvidia_run(query, root=None, dirty=True):
    calls = []
    git = git_answers(root, dirty) if root is not None else None

    def run(argv):
        calls.append(list(argv))
        if argv == ["nvidia-smi", "-L"]:
            return "GPU 0: redacted\n"
        if argv == ["nvidia-smi"]:
            return fx("nvidia_smi_header.txt")
        if argv[0] == "nvidia-smi" and argv[1].startswith("--query-gpu="):
            return query
        if argv[0] == "git" and git:
            return git(argv)
        return None

    run.calls = calls
    return run


def test_field_none_is_not_captured():
    assert field(None, "measured") == {"value": None, "source": "not_captured"}
    assert field(0, "measured") == {"value": 0, "source": "measured"}


def test_nvidia_host(tmp_path):
    repo = make_repo(tmp_path / "bakeoff")
    run = nvidia_run(fx("nvidia_query_1gpu.csv"), root=repo)
    env = capture_env(
        CFG,
        {"version": "0.9.1"},
        "served",
        run=run,
        read=fake_read(LINUX_FILES),
        platform="linux",
        repo_root=repo,
        identity=[],
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


def _env_with_repo(repo, run, **kw):
    return capture_env(
        CFG, None, None, run=run, read=fake_read(LINUX_FILES), platform="linux",
        repo_root=repo, identity=[], **kw,
    )  # fmt: skip


def test_no_git_gives_not_captured(tmp_path):
    repo = make_repo(tmp_path / "bakeoff")
    env = _env_with_repo(repo, nvidia_run(fx("nvidia_query_1gpu.csv")))  # git returns None
    assert env["harness"]["git_commit"]["source"] == "not_captured"
    assert env["harness"]["git_dirty"]["source"] == "not_captured"


def test_clean_tree(tmp_path):
    repo = make_repo(tmp_path / "bakeoff")
    env = _env_with_repo(repo, nvidia_run("", root=repo, dirty=False))
    assert env["harness"]["git_dirty"] == {"value": False, "source": "measured"}
    assert env["harness"]["git_commit"]["value"] == "0123456789abcdef0123456789abcdef01234567"


def test_git_not_captured_when_package_inside_unrelated_repo(tmp_path):
    # a venv/site-packages inside someone's private project: .git but no bakeoff pyproject
    other = tmp_path / "private-project"
    (other / ".git").mkdir(parents=True)
    (other / "pyproject.toml").write_text('[project]\nname = "someone-elses-app"\n')
    env = _env_with_repo(other, nvidia_run("", root=other))
    assert env["harness"]["git_commit"]["source"] == "not_captured"
    assert env["harness"]["git_dirty"]["source"] == "not_captured"
    bare = tmp_path / "no-git"
    bare.mkdir()
    (bare / "pyproject.toml").write_text('[project]\nname = "bakeoff"\n')
    env = _env_with_repo(bare, nvidia_run("", root=bare))
    assert env["harness"]["git_commit"]["source"] == "not_captured"


def test_git_not_captured_when_bakeoff_tree_nested_in_unrelated_repo(tmp_path):
    outer = tmp_path / "outer"
    inner = make_repo(outer / "vendor" / "bakeoff")
    inner_git = inner / ".git"
    inner_git.rmdir()  # a plain copy: no repo of its own, the toplevel is the outer repo
    (inner / ".git").write_text("gitdir: elsewhere\n")  # marker exists, toplevel disagrees
    env = _env_with_repo(inner, nvidia_run("", root=outer))
    assert env["harness"]["git_commit"]["source"] == "not_captured"
    assert env["harness"]["git_dirty"]["source"] == "not_captured"


def test_git_captured_for_worktree_style_git_file(tmp_path):
    repo = make_repo(tmp_path / "wt")
    (repo / ".git").rmdir()
    (repo / ".git").write_text("gitdir: /somewhere/else\n")
    env = _env_with_repo(repo, nvidia_run("", root=repo))
    assert env["harness"]["git_commit"]["source"] == "measured"


def test_git_path_not_recorded(tmp_path):
    repo = make_repo(tmp_path / "zz-distinctive-checkout-dir" / "bakeoff")
    run = nvidia_run(fx("nvidia_query_1gpu.csv"), root=repo)
    env = _env_with_repo(repo, run)
    git_calls = [c for c in run.calls if c[0] == "git"]
    assert any(str(repo) in c for c in git_calls)  # the path really was used...
    assert "zz-distinctive-checkout-dir" not in json.dumps(env)  # ...but never recorded
    assert env["harness"]["git_commit"]["source"] == "measured"


def test_only_allowlisted_commands_run(tmp_path):
    repo = make_repo(tmp_path / "bakeoff")
    run = nvidia_run(fx("nvidia_query_1gpu.csv"), root=repo)
    _env_with_repo(repo, run)
    git_tails = set()
    for argv in run.calls:
        assert argv[0] in {"nvidia-smi", "git"}
        if argv[0] == "nvidia-smi":
            assert len(argv) == 1 or argv[1] == "-L" or argv[1].startswith("--query-gpu=")
        else:
            assert argv[1] == "-C" and argv[2] == str(repo)
            git_tails.add(tuple(argv[3:]))
    assert git_tails == {
        ("rev-parse", "--show-toplevel"),
        ("rev-parse", "HEAD"),
        ("status", "--porcelain"),
    }
    mac = fake_run({("sw_vers",): "15.6.1\n", ("sysctl",): fx("sysctl_darwin.txt")})
    capture_env(CFG, None, None, run=mac, read=fake_read({}), platform="darwin", repo_root=repo)
    assert {c[0] for c in mac.calls} <= {"nvidia-smi", "sw_vers", "sysctl", "git"}


def test_unknown_platform_is_not_linux():
    env = capture_env(CFG, None, None, run=fake_run({}), read=fake_read(LINUX_FILES),
                      platform="win32")  # fmt: skip
    check_shape(env)
    assert all(v["source"] == "not_captured" for v in env["host"].values())
    assert set(env["host"]) == {"os", "os_version", "cpu", "cpu_cores", "ram_bytes"}
    assert env["gpus"] == []


def test_intel_mac_has_no_gpu_entry():
    sysctl = "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz\n16\n34359738368\n"
    run = fake_run({("sw_vers",): "14.5\n", ("sysctl",): sysctl})
    env = capture_env(CFG, None, None, run=run, read=fake_read({}), platform="darwin")
    assert env["gpus"] == []
    assert env["host"]["cpu"]["value"].startswith("Intel")


def test_free_text_is_capped_and_stripped():
    cpuinfo = "processor\t: 0\nmodel name\t: CPU\x1b[31m " + "x" * 300 + "\n"
    env = capture_env(CFG, {"version": "v\x00" + "9" * 200}, "m\n" + "y" * 200,
                      run=fake_run({}), read=fake_read({"/proc/cpuinfo": cpuinfo}),
                      platform="linux", identity=[])  # fmt: skip
    cpu = env["host"]["cpu"]["value"]
    assert len(cpu) == 100 and "\x1b" not in cpu
    assert len(env["runtime"]["version"]["value"]) == 100
    assert "\x00" not in env["runtime"]["version"]["value"]
    assert "\n" not in env["runtime"]["served_model"]["value"]


def test_free_text_containing_identity_is_not_captured():
    cpuinfo = "processor\t: 0\nmodel name\t: Xeon of lab-box-7\n"  # scrub-ok
    osrel = 'PRETTY_NAME="Ubuntu for alice"\n'  # scrub-ok
    env = capture_env(
        CFG,
        {"version": "build /home/alice/x"},  # scrub-ok
        "serving at 192.0.2.7",  # scrub-ok
        run=nvidia_run("0, GPU of lab-box-7, 100, 1.2, 10, 1, 1\n"),  # scrub-ok
        read=fake_read({"/proc/cpuinfo": cpuinfo, "/etc/os-release": osrel}),
        platform="linux",
        identity=["lab-box-7", "alice", "/home/alice", "xy"],  # scrub-ok
    )
    check_shape(env)
    assert env["host"]["cpu"]["source"] == "not_captured"
    assert env["host"]["os_version"]["source"] == "not_captured"
    assert env["gpus"][0]["name"]["source"] == "not_captured"
    assert env["runtime"]["version"]["source"] == "not_captured"
    assert env["runtime"]["served_model"]["source"] == "not_captured"
    # a term shorter than 3 chars is ignored
    ok = capture_env(CFG, None, "xy-model", run=fake_run({}), read=fake_read({}),
                     platform="linux", identity=["xy"])  # fmt: skip
    assert ok["runtime"]["served_model"]["value"] == "xy-model"


def test_default_identity_uses_hostname_user_home(monkeypatch):
    monkeypatch.setattr("bakeoff.env.socket.gethostname", lambda: "planted-host-9")
    monkeypatch.setattr("bakeoff.env.getpass.getuser", lambda: "planted-user")
    monkeypatch.setenv("HOME", "/srv/planted-home")
    for text in ("planted-host-9 cpu", "PLANTED-USER cpu", "at /srv/planted-home/x"):
        cpuinfo = f"processor\t: 0\nmodel name\t: {text}\n"
        env = capture_env(CFG, None, None, run=fake_run({}),
                          read=fake_read({"/proc/cpuinfo": cpuinfo}), platform="linux")  # fmt: skip
        assert env["host"]["cpu"]["source"] == "not_captured", text


def test_os_release_quotes_and_fallback():
    single = "PRETTY_NAME='Debian GNU/Linux 12 (bookworm)'\n"
    env = capture_env(CFG, None, None, run=fake_run({}),
                      read=fake_read({"/usr/lib/os-release": single}), platform="linux",
                      identity=[])  # fmt: skip
    assert env["host"]["os_version"]["value"] == "Debian GNU/Linux 12 (bookworm)"


def test_default_run_survives_non_utf8(tmp_path):
    from bakeoff.env import default_run

    script = tmp_path / "bad.py"
    script.write_bytes(b"import sys; sys.stdout.buffer.write(b'ok\\xff\\xfe')\n")
    out = default_run([sys.executable, str(script)])
    assert out is not None and out.startswith("ok")


def test_malformed_nvidia_row_is_kept_as_not_captured():
    csv_text = '0, "NVIDIA, Quoted Name", 8192, 570.1, 100.00, 1, 2\n1, Broken Row, 8192\n'
    rows = parse_nvidia_query(csv_text)
    assert rows[0]["name"] == "NVIDIA, Quoted Name"
    assert len(rows) == 2 and rows[1]["index"] == 1 and rows[1]["name"] is None
    env = capture_env(CFG, None, None, run=nvidia_run(csv_text), read=fake_read({}),
                      platform="linux", identity=[])  # fmt: skip
    assert len(env["gpus"]) == 2
    assert env["gpus"][1]["name"]["source"] == "not_captured"
    assert env["gpus"][1]["vram_mib"]["source"] == "not_captured"


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


def test_driver_string_goes_through_the_identity_cleaner():
    env = capture_env(
        CFG,
        None,
        "m",
        run=nvidia_run("0, GPU, 100, 1.2.3 lab-box-7, 10, 1, 1\n"),  # scrub-ok
        read=fake_read({}),
        platform="linux",
        identity=["lab-box-7"],  # scrub-ok
    )
    assert env["gpus"][0]["driver"] == {"value": None, "source": "not_captured"}
