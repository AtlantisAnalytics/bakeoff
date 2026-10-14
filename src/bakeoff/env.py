"""Allowlisted environment capture.

Only the fields named in the spec are recorded. Nothing here runs a command that prints
identity (hostname, uname -n, nvidia-smi -q, ioreg, system_profiler), and `nvidia-smi -L` is used
for detection only because its output carries GPU UUIDs.
"""

import csv
import platform as _platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from bakeoff import __version__
from bakeoff.config import RunConfig

Runner = Callable[[list[str]], str | None]
Reader = Callable[[str], str | None]

NVIDIA_QUERY = (
    "--query-gpu=index,name,memory.total,driver_version,power.limit,memory.used,power.draw"
)
NVIDIA_QUERY_ARGV = ["nvidia-smi", NVIDIA_QUERY, "--format=csv,noheader,nounits"]


def default_run(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def default_read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def field(value: object, source: str) -> dict:
    if value is None:
        return {"value": None, "source": "not_captured"}
    return {"value": value, "source": source}


def _num(text: str, cast: Callable) -> object:
    text = text.strip()
    if not text or text.startswith("["):  # "[N/A]", "[Not Supported]"
        return None
    try:
        return cast(text)
    except ValueError:
        return None


def parse_nvidia_query(text: str) -> list[dict]:
    rows = []
    for rec in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(rec) != 7:
            continue
        idx, name, vram, driver, limit, used, draw = (c.strip() for c in rec)
        rows.append(
            {
                "index": _num(idx, int),
                "name": name or None,
                "vram_mib": _num(vram, int),
                "driver": None if driver.startswith("[") or not driver else driver,
                "power_limit_w": _num(limit, float),
                "mem_used_mib": _num(used, int),
                "power_w": _num(draw, float),
            }
        )
    return rows


def parse_cuda_version(nvidia_smi_text: str) -> str | None:
    m = re.search(r"CUDA Version:\s*(\d+(?:\.\d+)*)", nvidia_smi_text)
    return m.group(1) if m else None


def parse_vm_stat(text: str) -> int | None:
    head = re.search(r"page size of (\d+) bytes", text)
    if not head:
        return None
    pages = 0
    for label in ("Pages wired down", "Pages active", "Pages occupied by compressor"):
        m = re.search(rf"^{label}:\s+(\d+)\.", text, re.M)
        if not m:
            return None
        pages += int(m.group(1))
    return pages * int(head.group(1))


def detect_platform(run: Runner, platform: str = sys.platform) -> str:
    if run(["nvidia-smi", "-L"]) is not None:  # output is never recorded: it holds GPU UUIDs
        return "nvidia"
    return "darwin" if platform == "darwin" else "none"


def _os_release_name(text: str | None) -> str | None:
    m = re.search(r'^PRETTY_NAME="?([^"\n]*)"?\s*$', text or "", re.M)
    return m.group(1).strip() or None if m else None


def _cpuinfo(text: str | None) -> tuple[str | None, int | None]:
    text = text or ""
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    cores = len(re.findall(r"^processor\s*:", text, re.M))
    return (m.group(1).strip() if m else None, cores or None)


def _meminfo_bytes(text: str | None) -> int | None:
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text or "", re.M)
    return int(m.group(1)) * 1024 if m else None


def _int_or_none(text: str) -> int | None:
    return int(text) if text.strip().isdigit() else None


def _linux_host(read: Reader) -> dict:
    cpu, cores = _cpuinfo(read("/proc/cpuinfo"))
    return {
        "os": field("Linux", "measured"),
        "os_version": field(_os_release_name(read("/etc/os-release")), "measured"),
        "cpu": field(cpu, "measured"),
        "cpu_cores": field(cores, "measured"),
        "ram_bytes": field(_meminfo_bytes(read("/proc/meminfo")), "measured"),
    }


def _mac_sysctl(run: Runner) -> tuple[str | None, int | None, int | None]:
    out = run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"])
    lines = [ln.strip() for ln in (out or "").splitlines()]
    if len(lines) != 3:
        return None, None, None
    return lines[0] or None, _int_or_none(lines[1]), _int_or_none(lines[2])


def _mac_host(run: Runner, sysctl: tuple[str | None, int | None, int | None]) -> dict:
    version = (run(["sw_vers", "-productVersion"]) or "").strip() or None
    cpu, cores, mem = sysctl
    return {
        "os": field("macOS", "measured"),
        "os_version": field(version, "measured"),
        "cpu": field(cpu, "measured"),
        "cpu_cores": field(cores, "measured"),
        "ram_bytes": field(mem, "measured"),
    }


def _nvidia_gpus(run: Runner) -> list[dict]:
    rows = parse_nvidia_query(run(NVIDIA_QUERY_ARGV) or "")
    cuda = parse_cuda_version(run(["nvidia-smi"]) or "")
    return [
        {
            "name": field(r["name"], "measured"),
            "vram_mib": field(r["vram_mib"], "measured"),
            "driver": field(r["driver"], "measured"),
            "cuda": field(cuda, "measured"),
            "power_limit_w": field(r["power_limit_w"], "measured"),
        }
        for r in rows
    ]


def _git(run: Runner) -> tuple[str | None, bool | None]:
    pkg = str(Path(__file__).resolve().parent)  # used to run git; never recorded
    head = run(["git", "-C", pkg, "rev-parse", "HEAD"])
    commit = (head or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None, None
    status = run(["git", "-C", pkg, "status", "--porcelain"])
    return commit, None if status is None else bool(status.strip())


def capture_env(
    cfg: RunConfig | None,
    runtime: dict | None,
    served_model: str | None,
    run: Runner = default_run,
    read: Reader = default_read,
    platform: str = sys.platform,
) -> dict:
    kind = detect_platform(run, platform)
    mac = platform == "darwin"
    sysctl = _mac_sysctl(run) if mac else (None, None, None)
    host = _mac_host(run, sysctl) if mac else _linux_host(read)
    if kind == "nvidia":
        gpus = _nvidia_gpus(run)
    elif kind == "darwin":
        gpus = [
            {
                "name": field(sysctl[0], "measured"),
                "unified_memory_bytes": field(sysctl[2], "measured"),
            }
        ]
    else:
        gpus = []
    model = cfg.model if cfg else None
    commit, dirty = _git(run)
    return {
        "host": host,
        "gpus": gpus,
        "runtime": {
            "name": field(model.runtime if model else None, "operator"),
            "version": field((runtime or {}).get("version"), "reported"),
            "served_model": field(served_model, "reported"),
        },
        "model": {
            "label": field(model.label if model else None, "operator"),
            "weights": field(model.weights if model else None, "operator"),
            "weights_file": field((model.weights_file or None) if model else None, "operator"),
            "quant": field((model.quant or None) if model else None, "operator"),
            "envelope": field((model.envelope or None) if model else None, "operator"),
        },
        "harness": {
            "version": field(__version__, "measured"),
            "git_commit": field(commit, "measured"),
            "git_dirty": field(dirty, "measured"),
            "python": field(_platform.python_version(), "measured"),
        },
    }
