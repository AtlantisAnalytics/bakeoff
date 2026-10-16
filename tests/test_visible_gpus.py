"""A run on a multi-GPU host records and samples only its own GPUs (CUDA_VISIBLE_DEVICES)."""

import pytest

from bakeoff.env import NVIDIA_QUERY_ARGV, _nvidia_gpus, nvidia_argv
from bakeoff.sampler import SAMPLER_NVIDIA_ARGV, Sampler

ROW = "2, NVIDIA GeForce RTX 3060, 12288, 570.207, 170.00, 9000, 120.5\n"


def test_no_visible_devices_queries_every_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert nvidia_argv(NVIDIA_QUERY_ARGV) == NVIDIA_QUERY_ARGV


@pytest.mark.parametrize(
    "raw, ids",
    [
        ("2", "2"),
        ("0,2", "0,2"),
        (" 1 , 3 ", "1,3"),
        ("GPU-8f2c1a0e-1111-2222-3333-444455556666", "GPU-8f2c1a0e-1111-2222-3333-444455556666"),
    ],
)
def test_visible_devices_become_nvidia_smi_ids(monkeypatch, raw, ids):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
    assert nvidia_argv(NVIDIA_QUERY_ARGV) == ["nvidia-smi", "-i", ids, *NVIDIA_QUERY_ARGV[1:]]


@pytest.mark.parametrize("raw", ["", " ", "2;rm -rf /", "-1", "2,,3", "all"])
def test_unusable_visible_devices_fall_back_to_every_gpu(monkeypatch, raw):
    """Never hand nvidia-smi something it would misread; record every GPU instead."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
    assert nvidia_argv(SAMPLER_NVIDIA_ARGV) == SAMPLER_NVIDIA_ARGV


def test_env_capture_records_only_visible_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    seen = []

    def run(argv):
        seen.append(argv)
        return ROW if "--query-gpu" in " ".join(argv) else "CUDA Version: 12.8"

    gpus = _nvidia_gpus(run, lambda v: v)
    assert ["nvidia-smi", "-i", "2", *NVIDIA_QUERY_ARGV[1:]] in seen
    assert [g["name"]["value"] for g in gpus] == ["NVIDIA GeForce RTX 3060"]


def test_sampler_samples_only_visible_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    seen = []

    def run(argv):
        seen.append(argv)
        return "2, 9000, 120.5\n"

    rec = Sampler("nvidia", run).sample_once()
    assert seen == [["nvidia-smi", "-i", "2", *SAMPLER_NVIDIA_ARGV[1:]]]
    assert rec["gpus"] == [{"index": 2, "mem_used_mib": 9000, "power_w": 120.5}]
