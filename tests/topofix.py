"""Fixture builder for the topology page: bundle.json files with only the fields the page
reads, an experiment.toml and a site_src fragment."""

import json

from bakeoff.web.topology import EXPERIMENT

RUNS = f"{EXPERIMENT}/runs"


def bundle(
    root,
    rel,
    *,
    envelope="2× RTX 3060 12 GB",
    parallel="tp=2",
    per_gpu=(9000, 9000),
    paths=("NODE",),
    dec=79.06,
    agg=280.12,
    ttft=0.0331,
    ctx=32768,
    power=245.4,
):
    d = root / rel
    d.mkdir(parents=True)
    peak = {"mem_used_mib_per_gpu": list(per_gpu), "power_w_mean": power}
    meta = {
        "status": "complete",
        "config": {"model": {"envelope": envelope, "parallel": parallel, "runtime": "vllm"}},
        "env": {
            "gpus": [{} for _ in per_gpu],
            "gpu_links": [{"path": {"value": p, "source": "measured"}} for p in paths],
        },
        "speed": {
            "short@1": {"decode_tps": {"median": dec}, "ttft_s": {"median": ttft}},
            "short@4": {"aggregate_tps": {"median": agg}},
        },
        "fit": {"max_context_passed": ctx},
        "memory": {"peaks": {"value": {"baseline": peak, "speed": peak}, "source": "measured"}},
    }
    (d / "bundle.json").write_text(json.dumps(meta), encoding="utf-8")
    return rel


def write_experiment(root, groups, page):
    """groups: {key: [bundle rel paths]}; page: the site_src fragment."""
    lines = []
    for key, rels in groups.items():
        lines += ["[[group]]", f'key = "{key}"', "bundles = ["]
        lines += [f'  "{r}",' for r in rels] + ["]", ""]
    exp = root / EXPERIMENT
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "experiment.toml").write_text("\n".join(lines), encoding="utf-8")
    (root / "site_src").mkdir(exist_ok=True)
    (root / "site_src" / "topology.html").write_text(page, encoding="utf-8")
