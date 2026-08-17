"""Configuration: dataclass defaults <- YAML file <- CLI overrides.

Method presets map paper names to (edge_weighting, central_weighting):
  bl1  = uniform     + uniform
  bl2  = samplesize  + uniform
  escs = samplesize  + avg_samplesize
  esca = samplesize  + accuracy
  auto = samplesize  + divergence-aware selector (ESCS vs ESCA via tau)
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

METHOD_PRESETS = {
    "bl1": ("uniform", "uniform"),
    "bl2": ("samplesize", "uniform"),
    "escs": ("samplesize", "avg_samplesize"),
    "esca": ("samplesize", "accuracy"),
    "auto": ("samplesize", "auto"),
}


@dataclass
class Config:
    # experiment identity
    name: str = "run"
    seed: int = 42
    outdir: str = "runs"
    device: str = "auto"          # auto | cpu | cuda

    # data
    dataset: str = "synthetic"    # synthetic | mnist | fmnist | cifar10
    data_root: str = "./data"
    partition: str = "pathological"  # iid|pathological|dirichlet|powerlaw|compound
    classes_per_client: int = 2
    dirichlet_alpha: float = 0.5
    powerlaw_exponent: float = 1.2
    val_fraction: float = 0.2     # server-side public validation split (of test)

    # topology
    num_clients: int = 100
    num_edges: int = 20
    feasible_edges: int = 0       # 0 => unconstrained; else |F_k|

    # schedule
    global_rounds: int = 50       # T
    intermediate_rounds: int = 2  # U
    local_epochs: int = 6         # E
    client_frac: float = 0.5      # client participation per intermediate round
    edge_frac: float = 1.0        # edge participation per global round

    # optimization
    batch_size: int = 20
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    eta: float = 1.0              # edge lr on deltas (cp-np)

    # training DP
    dp_mode: str = "cp-np"        # baseline | cg-ng | cg-np | cp-np
    clip: float = 1.0
    sigma: float = 1.0            # noise std on clipped payload
    delta: float = 1e-5

    # method (aggregation weighting)
    method: str = "escs"          # bl1|bl2|escs|esca|auto (sets fields below)
    edge_weighting: str = "samplesize"
    central_weighting: str = "avg_samplesize"
    tau: float = 0.15             # selector threshold on D_res (L1)

    # assignment
    assignment: str = "random"    # random | oracle | nonprivate | rr | hdpba
    eps_assign: float = 0.2       # eps_a per exponential-mechanism pass
    eps_agg: float = 0.1          # per aggregate release
    delta_agg: float = 1e-6
    assign_passes: int = 2        # R
    assign_record_level: bool = True

    # bookkeeping
    log_every: int = 1
    config_path: Optional[str] = None

    def resolve(self) -> "Config":
        if self.method in METHOD_PRESETS:
            self.edge_weighting, self.central_weighting = METHOD_PRESETS[self.method]
        return self


def load_config(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="H-DPBA + DP-HFL")
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    known, unknown = ap.parse_known_args(argv)

    cfg = Config()
    if known.config:
        import yaml
        with open(known.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key in YAML: {k}")
            setattr(cfg, k, v)
        cfg.config_path = known.config

    # CLI overrides: --key value for any dataclass field
    ov = argparse.ArgumentParser()
    for f_ in dataclasses.fields(Config):
        if f_.name == "config_path":
            continue
        typ = f_.type if isinstance(f_.type, type) else type(getattr(cfg, f_.name))
        if typ is bool:
            ov.add_argument(f"--{f_.name}", type=lambda s: s.lower() in
                            ("1", "true", "yes"), default=None)
        else:
            ov.add_argument(f"--{f_.name}", type=typ, default=None)
    over, _ = ov.parse_known_args(unknown)
    for k, v in vars(over).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg.resolve()
