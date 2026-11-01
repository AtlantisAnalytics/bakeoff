"""Allowlisted environment capture.

Only the fields named in the spec are recorded. Nothing here runs a command that prints
identity (hostname, uname -n, nvidia-smi -q, ioreg, system_profiler), and `nvidia-smi -L` is used
for detection only because its output carries GPU UUIDs. PCI bus ids are read only to find each
GPU's NUMA node in sysfs and to match it against gpu_unlock's targets; they are never recorded.
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
LINK_QUERY_ARGV = [
    "nvidia-smi",
    "--query-gpu=index,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,pcie.link.width.current",
    "--format=csv,noheader,nounits",
]
TOPO_ARGV = ["nvidia-smi", "topo", "-m"]
P2P_ARGV = ["nvidia-smi", "topo", "-p2p", "r"]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BUS = re.compile(r"([0-9A-Fa-f]{1,8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])")
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


def parse_link_query(text: str) -> dict[int, dict]:
    """``LINK_QUERY_ARGV`` output, keyed by nvidia-smi index; malformed rows are dropped."""
    rows = {}
    for rec in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(rec) != 5:
            continue
        idx, bus, gen, width_max, width = (c.strip() for c in rec)
        index = _num(idx, int)
        if index is None:
            continue
        rows[index] = {
            "bus": bus if _BUS.fullmatch(bus) else None,
            "gen_max": _num(gen, int),
            "width_max": _num(width_max, int),
            "width": _num(width, int),
        }
    return rows


def parse_topo_matrix(text: str) -> dict[tuple[int, int], str]:
    """The GPU-to-GPU cells of ``nvidia-smi topo -m`` or ``topo -p2p r``, keyed (i, j) with
    i < j on nvidia-smi indices. CPU affinity and NIC columns are ignored."""
    cols: dict[int, int] = {}
    cells: dict[tuple[int, int], str] = {}
    for line in _ANSI.sub("", text).splitlines():
        tokens = [t.strip() for t in line.split("\t")]
        if not cols and tokens and tokens[0] == "" and "GPU0" in tokens:
            cols = {pos: int(t[3:]) for pos, t in enumerate(tokens) if re.fullmatch(r"GPU\d+", t)}
            continue
        m = re.fullmatch(r"GPU(\d+)", tokens[0]) if cols and tokens else None
        if not m:
            continue
        i = int(m.group(1))
        for pos, j in cols.items():
            if i < j and pos < len(tokens) and re.fullmatch(r"[A-Z0-9#]{1,8}", tokens[pos]):
                cells[(i, j)] = tokens[pos]
    return cells


def _pci_name(bus: str | None) -> str | None:
    """A bus id as sysfs and `lspci -D` name it: "00000000:AF:00.0" -> "0000:af:00.0"."""
    m = _BUS.fullmatch((bus or "").strip())
    if not m:
        return None
    dom, b, d, f = m.groups()
    return f"{int(dom, 16):04x}:{b.lower()}:{d.lower()}.{f}"


def _numa_node(read: Reader, bus: str | None) -> int | None:
    name = _pci_name(bus)
    if name is None:
        return None
    node = _num(read(f"/sys/bus/pci/devices/{name}/numa_node") or "", int)
    return node if isinstance(node, int) and node >= 0 else None  # -1: no NUMA information


# gpu_unlock (github.com/Zagitta/gpu-unlock) lifts GeForce's half-rate cap on FP16/BF16/TF32
# tensor math with FP32 accumulation at boot. Prefill and TTFT change with it; decode doesn't.
UNLOCK_MODULE = "/sys/module/gpu_unlock"


def _unlock_targets(read: Reader) -> str | frozenset[str] | None:
    """ "off" when the module isn't loaded, "all", the targeted PCI names, or None (unknown)."""
    if (read(f"{UNLOCK_MODULE}/initstate") or "").strip() != "live":
        return "off"
    raw = read(f"{UNLOCK_MODULE}/parameters/targets")
    if raw is None:
        cmdline = read("/proc/cmdline")
        if cmdline is None:
            return None
        given = [t.split("=", 1)[1] for t in cmdline.split() if t.startswith("gpu_unlock.targets=")]
        raw = given[-1] if given else ""
    if raw.strip() == "all":
        return "all"
    # the module ignores "all" mixed with addresses, and anything else it can't parse
    return frozenset(n for n in (_pci_name(t) for t in raw.split(",")) if n)


def _tensor_unlock(targets: str | frozenset[str] | None, bus: str | None) -> str | None:
    """ "targeted" says the module was asked to unlock this card, not that the unlock took."""
    if targets is None or targets == "off":
        return targets
    if targets == "all":
        return "targeted"
    if not targets:
        return "loaded"
    name = _pci_name(bus)
    if name is None:
        return None
    return "targeted" if name in targets else "loaded"


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


def _pmset_thermal(text: str | None) -> int | None:
    if text is None:
        return None
    if "No thermal warning level has been recorded" in text:
        return 0
    m = re.search(r"thermal warning level\D*(\d+)", text, re.I)
    return int(m.group(1)) if m else None


def mac_thermal_warning(run: Runner) -> dict:
    return field(_pmset_thermal(run(["pmset", "-g", "therm"])), "measured")


def _low_power_mode(text: str | None) -> bool | None:
    """lowpowermode if listed, else powermode (0 automatic, 1 low power, 2 high power)."""
    for key in ("lowpowermode", "powermode"):
        m = re.search(rf"^\s*{key}\s+(\d+)\s*$", text or "", re.M)
        if m:
            return int(m.group(1)) == 1
    return None


def mac_power_state(run: Runner) -> dict:
    """Power source, Low Power Mode, thermal warning and GPU wired-memory limit."""
    src = re.search(r"Now drawing from '([^']+)'", run(["pmset", "-g", "batt"]) or "")
    limit = _int_or_none(run(["sysctl", "-n", "iogpu.wired_limit_mb"]) or "")
    return {
        "power_source": field(src.group(1) if src else None, "measured"),
        "low_power_mode": field(_low_power_mode(run(["pmset", "-g"])), "measured"),
        "thermal_warning": mac_thermal_warning(run),
        "gpu_wired_limit_mb": field(limit, "measured"),
    }


def _nvidia_gpus(
    run: Runner, read: Reader, clean: Cleaner, linux: bool
) -> tuple[list[dict], list[dict]]:
    """The run's GPUs, and one link entry per pair of them (numbered by position in the list)."""
    rows = parse_nvidia_query(run(nvidia_argv(NVIDIA_QUERY_ARGV)) or "")
    cuda = parse_cuda_version(run(["nvidia-smi"]) or "")
    links = parse_link_query(run(nvidia_argv(LINK_QUERY_ARGV)) or "")
    unlock = _unlock_targets(read) if linux else None
    gpus = []
    for r in rows:
        link = links.get(r["index"], {})
        gpus.append(
            {
                "name": field(clean(r["name"]), "measured"),
                "vram_mib": field(r["vram_mib"], "measured"),
                "driver": field(clean(r["driver"]), "measured"),
                "cuda": field(cuda, "measured"),
                "power_limit_w": field(r["power_limit_w"], "measured"),
                "pcie_gen_max": field(link.get("gen_max"), "measured"),
                "pcie_width_max": field(link.get("width_max"), "measured"),
                # the width the link trained to; the current generation is not recorded because
                # idle GPUs drop to gen 1 between requests
                "pcie_width": field(link.get("width"), "measured"),
                "numa_node": field(_numa_node(read, link.get("bus")), "measured"),
                "tensor_unlock": field(_tensor_unlock(unlock, link.get("bus")), "measured"),
            }
        )
    if len(rows) < 2:
        return gpus, []
    path = parse_topo_matrix(run(TOPO_ARGV) or "")
    p2p = parse_topo_matrix(run(P2P_ARGV) or "")
    pairs = []
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            ia, ib = rows[a]["index"], rows[b]["index"]
            key = (min(ia, ib), max(ia, ib)) if ia is not None and ib is not None else None
            pairs.append(
                {
                    "gpus": field([a, b], "measured"),
                    "path": field(path.get(key), "measured"),
                    "p2p_read": field(p2p.get(key), "measured"),
                }
            )
    return gpus, pairs


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


def _runtime_version(reported: str | None, model) -> dict:
    """The version the endpoint reports, else the operator's (ModelConfig.runtime_version)."""
    if reported is not None or not (model and model.runtime_version):
        return field(reported, "reported")
    return field(model.runtime_version, "operator")


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
        host["power_state"] = mac_power_state(run)
    elif platform.startswith("linux"):
        host = _linux_host(read, clean)
    else:
        host = {k: field(None, "measured") for k in _HOST_KEYS}
    links: list[dict] = []
    if kind == "nvidia":
        gpus, links = _nvidia_gpus(run, read, clean, platform.startswith("linux"))
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
        "gpu_links": links,
        "runtime": {
            "name": field(model.runtime if model else None, "operator"),
            "version": _runtime_version(clean((runtime or {}).get("version")), model),
            "served_model": field(clean(served_model), "reported"),
        },
        "model": {
            "label": field(model.label if model else None, "operator"),
            "weights": field(model.weights if model else None, "operator"),
            "weights_file": field((model.weights_file or None) if model else None, "operator"),
            "quant": field((model.quant or None) if model else None, "operator"),
            "envelope": field((model.envelope or None) if model else None, "operator"),
            "parallel": field((model.parallel or None) if model else None, "operator"),
        },
        "harness": {
            "version": field(__version__, "measured"),
            "git_commit": field(commit, git_source),
            "git_dirty": field(dirty, git_source),
            "python": field(_platform.python_version(), "measured"),
        },
    }
