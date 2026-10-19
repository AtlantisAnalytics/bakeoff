"""Allowlisted environment capture.

Only the fields named in the spec are recorded. Nothing here runs a command that prints
identity (hostname, uname -n, nvidia-smi -q, ioreg, system_profiler), and `nvidia-smi -L` is used
for detection only because its output carries GPU UUIDs.
"""

import csv
import getpass
import os
import platform as _platform
import re
import socket
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

from bakeoff import __version__
from bakeoff.config import RunConfig

Runner = Callable[[list[str]], str | None]
Reader = Callable[[str], str | None]
Cleaner = Callable[[object], str | None]
_HOST_KEYS = ("os", "os_version", "cpu", "cpu_cores", "ram_bytes")

NVIDIA_QUERY = (
    "--query-gpu=index,name,memory.total,driver_version,power.limit,memory.used,power.draw"
)
NVIDIA_QUERY_ARGV = ["nvidia-smi", NVIDIA_QUERY, "--format=csv,noheader,nounits"]
_GPU_ID = re.compile(r"\d+|GPU-[0-9A-Fa-f-]+|MIG-[0-9A-Za-z/-]+")


def nvidia_argv(argv: Sequence[str]) -> list[str]:
    """``argv`` limited to the GPUs in CUDA_VISIBLE_DEVICES (``nvidia-smi -i``), so a run on a
    multi-GPU host records and samples only the GPUs it was given. Unset, empty or unparseable:
    every GPU, as before. The ids go to nvidia-smi only; they are never recorded."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [t.strip() for t in raw.split(",")]
    if not raw.strip() or not all(_GPU_ID.fullmatch(t) for t in ids):
        return list(argv)
    return [argv[0], "-i", ",".join(ids), *argv[1:]]


def default_run(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=10)
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
        if not rec:
            continue
        if len(rec) != 7:  # malformed: keep the GPU, with every field not_captured
            idx = rec[0].strip()
            rows.append(
                {
                    "index": _num(idx, int),
                    "name": None,
                    "vram_mib": None,
                    "driver": None,
                    "power_limit_w": None,
                    "mem_used_mib": None,
                    "power_w": None,
                }
            )
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
    m = re.search(r"^PRETTY_NAME=(.*)$", text or "", re.M)
    if not m:
        return None
    value = m.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip() or None


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


def _linux_host(read: Reader, clean: Cleaner) -> dict:
    cpu, cores = _cpuinfo(read("/proc/cpuinfo"))
    name = _os_release_name(read("/etc/os-release")) or _os_release_name(
        read("/usr/lib/os-release")
    )
    return {
        "os": field("Linux", "measured"),
        "os_version": field(clean(name), "measured"),
        "cpu": field(clean(cpu), "measured"),
        "cpu_cores": field(cores, "measured"),
        "ram_bytes": field(_meminfo_bytes(read("/proc/meminfo")), "measured"),
    }


def _mac_sysctl(run: Runner) -> tuple[str | None, int | None, int | None]:
    out = run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"])
    lines = [ln.strip() for ln in (out or "").splitlines()]
    if len(lines) != 3:
        return None, None, None
    return lines[0] or None, _int_or_none(lines[1]), _int_or_none(lines[2])


def _mac_host(
    run: Runner, sysctl: tuple[str | None, int | None, int | None], clean: Cleaner
) -> dict:
    version = (run(["sw_vers", "-productVersion"]) or "").strip() or None
    cpu, cores, mem = sysctl
    return {
        "os": field("macOS", "measured"),
        "os_version": field(clean(version), "measured"),
        "cpu": field(clean(cpu), "measured"),
        "cpu_cores": field(cores, "measured"),
        "ram_bytes": field(mem, "measured"),
    }


def _nvidia_gpus(run: Runner, clean: Cleaner) -> list[dict]:
    rows = parse_nvidia_query(run(nvidia_argv(NVIDIA_QUERY_ARGV)) or "")
    cuda = parse_cuda_version(run(["nvidia-smi"]) or "")
    return [
        {
            "name": field(clean(r["name"]), "measured"),
            "vram_mib": field(r["vram_mib"], "measured"),
            "driver": field(clean(r["driver"]), "measured"),
            "cuda": field(cuda, "measured"),
            "power_limit_w": field(r["power_limit_w"], "measured"),
        }
        for r in rows
    ]


def _is_bakeoff_root(root: Path) -> bool:
    if not (root / ".git").exists():
        return False
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text())
    except (OSError, ValueError):
        return False
    return data.get("project", {}).get("name") == "bakeoff"


def _git(run: Runner, root: Path) -> tuple[str | None, bool | None]:
    """Commit and dirty flag, only when `root` is itself the bakeoff checkout."""
    if not _is_bakeoff_root(root):
        return None, None
    top = (run(["git", "-C", str(root), "rev-parse", "--show-toplevel"]) or "").strip()
    try:
        if not top or Path(top).resolve() != root.resolve():  # compared only, never recorded
            return None, None
    except OSError:
        return None, None
    commit = (run(["git", "-C", str(root), "rev-parse", "HEAD"]) or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None, None
    status = run(["git", "-C", str(root), "status", "--porcelain"])
    return commit, None if status is None else bool(status.strip())


def _archive_commit(read: Reader, path: Path) -> str | None:
    """The commit `git archive` wrote into `_commit` (export-subst); None for the placeholder."""
    text = (read(str(path)) or "").strip()
    return text if re.fullmatch(r"[0-9a-f]{40}", text) else None


_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def default_identity() -> list[str]:
    """Strings that identify this machine or user; used only to reject free text."""
    terms = [os.environ.get("HOME", "")]
    try:
        terms.append(socket.gethostname())
    except OSError:
        pass
    try:
        terms.append(getpass.getuser())
    except (OSError, KeyError, ImportError):
        pass
    return terms


def make_cleaner(identity: Sequence[str]) -> Cleaner:
    terms = [t.lower() for t in identity if t and len(t) >= 3]

    def clean(value: object) -> str | None:
        if value is None:
            return None
        text = _CONTROL.sub("", str(value)).strip()
        low = text.lower()
        if not text or _IPV4.search(text) or any(t in low for t in terms):
            return None
        return text[:100]

    return clean


def capture_env(
    cfg: RunConfig | None,
    runtime: dict | None,
    served_model: str | None,
    run: Runner = default_run,
    read: Reader = default_read,
    platform: str = sys.platform,
    repo_root: Path | None = None,
    identity: Sequence[str] | None = None,
    commit_file: Path | None = None,
) -> dict:
    clean = make_cleaner(default_identity() if identity is None else identity)
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    kind = detect_platform(run, platform)
    mac = platform == "darwin"
    sysctl = _mac_sysctl(run) if mac else (None, None, None)
    if mac:
        host = _mac_host(run, sysctl, clean)
    elif platform.startswith("linux"):
        host = _linux_host(read, clean)
    else:
        host = {k: field(None, "measured") for k in _HOST_KEYS}
    if kind == "nvidia":
        gpus = _nvidia_gpus(run, clean)
    elif kind == "darwin" and (sysctl[0] or "").startswith("Apple"):
        gpus = [
            {
                "name": field(clean(sysctl[0]), "measured"),
                "unified_memory_bytes": field(sysctl[2], "measured"),
            }
        ]
    else:
        gpus = []
    model = cfg.model if cfg else None
    commit, dirty = _git(run, root)
    git_source = "measured"
    if commit is None:  # no checkout: a harness unpacked from `git archive` names its commit
        archived = _archive_commit(read, commit_file or Path(__file__).with_name("_commit"))
        if archived is not None:
            commit, dirty, git_source = archived, False, "reported"
    return {
        "host": host,
        "gpus": gpus,
        "runtime": {
            "name": field(model.runtime if model else None, "operator"),
            "version": field(clean((runtime or {}).get("version")), "reported"),
            "served_model": field(clean(served_model), "reported"),
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
            "git_commit": field(commit, git_source),
            "git_dirty": field(dirty, git_source),
            "python": field(_platform.python_version(), "measured"),
        },
    }
