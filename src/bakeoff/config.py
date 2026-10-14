"""Loads and validates run.toml plus CLI overrides."""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUNTIMES = ("vllm", "llamacpp", "ollama", "mlx", "other")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    model: str
    api_key_env: str = ""
    timeout_s: float = 600.0


@dataclass(frozen=True)
class ModelConfig:
    label: str
    weights: str
    weights_file: str = ""
    quant: str = ""
    runtime: str = "other"
    envelope: str = ""


@dataclass(frozen=True)
class QualityConfig:
    suites: list[str] = field(default_factory=list)
    private_suites: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SpeedConfig:
    concurrency: list[int] = field(default_factory=lambda: [1, 4])
    repeats: int = 5
    warmup: int = 1


@dataclass(frozen=True)
class FitConfig:
    steps: list[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768, 65536, 131072])
    max_context: int = 0


@dataclass(frozen=True)
class RunConfig:
    endpoint: EndpointConfig
    model: ModelConfig
    quality: QualityConfig
    speed: SpeedConfig
    fit: FitConfig


_SECTIONS: dict[str, type] = {
    "endpoint": EndpointConfig,
    "model": ModelConfig,
    "quality": QualityConfig,
    "speed": SpeedConfig,
    "fit": FitConfig,
}
_REQUIRED = {
    "endpoint": ("base_url", "model"),
    "model": ("label", "weights"),
}


def _parse_override_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_overrides(data: dict[str, dict[str, Any]], overrides: list[str]) -> None:
    for item in overrides:
        key, sep, raw = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot or not name:
            raise ConfigError(f"bad override {item!r}: expected section.key=value")
        if section not in _SECTIONS:
            raise ConfigError(f"bad override {item!r}: unknown section {section!r}")
        data.setdefault(section, {})[name] = _parse_override_value(raw.strip())


def _check_type(where: str, value: Any, default: Any) -> None:
    if isinstance(default, bool) or default is dataclasses.MISSING:
        return
    if isinstance(default, float):
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, str):
        ok = isinstance(value, str)
    elif isinstance(default, list):
        ok = isinstance(value, list)
    else:
        ok = True
    if not ok:
        raise ConfigError(f"{where} has the wrong type: {value!r}")


def _build_section(name: str, cls: type, table: dict[str, Any]) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(table) - set(fields))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(unknown)}")
    for req in _REQUIRED.get(name, ()):
        if req not in table or table[req] in ("", None):
            raise ConfigError(f"missing required key: {name}.{req}")
    kwargs = {}
    for key, value in table.items():
        f = fields[key]
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            default = ""
        _check_type(f"{name}.{key}", value, default)
        kwargs[key] = float(value) if isinstance(default, float) else value
    return cls(**kwargs)


def load_config(path: Path, overrides: list[str] = []) -> RunConfig:  # noqa: B006
    try:
        with Path(path).open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from e
    unknown = sorted(set(data) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")
    _apply_overrides(data, overrides)
    built = {name: _build_section(name, cls, data.get(name, {})) for name, cls in _SECTIONS.items()}
    if built["model"].runtime not in RUNTIMES:
        raise ConfigError(
            f"model.runtime {built['model'].runtime!r} is not allowed; use one of: "
            f"{', '.join(RUNTIMES)}"
        )
    return RunConfig(**built)


def public_config(cfg: RunConfig) -> dict:
    d = dataclasses.asdict(cfg)
    d["endpoint"].pop("base_url", None)
    d["endpoint"].pop("api_key_env", None)
    return d
