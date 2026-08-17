"""Client / Edge / Cloud components for three-tier DP-HFL.

Client DP modes:
  baseline : plain local SGD
  cg-ng    : per-batch gradient L2-clipping + per-batch Gaussian noise on the
             clipped gradient (DP-SGD-style local perturbation).
  cg-np    : per-batch global-L2 gradient clipping; Gaussian noise added to
             model parameters after local epochs, before upload
  cp-np    : train normally; clip the parameter DELTA (global L2); add
             Gaussian noise to the clipped delta; upload the noisy delta only

Edge weighting  : uniform | samplesize          (client tier)
Cloud weighting : uniform (BL1/BL2) | avg_samplesize (ESCS) |
                  accuracy (ESCA, server-side public validation set) |
                  auto (divergence-aware selector: D_res vs tau)
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# helpers on state dicts
# -----------------------------------------------------------------------------

def sd_zeros_like(sd):
    return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sd.items()}


def sd_add_(acc, sd, w: float):
    for k in acc:
        acc[k] += sd[k].to(torch.float32) * w


def sd_weighted_average(sds: List[dict], weights: List[float]) -> dict:
    ws = np.asarray(weights, dtype=np.float64)
    ws = ws / ws.sum()
    out = sd_zeros_like(sds[0])
    for sd, w in zip(sds, ws):
        sd_add_(out, sd, float(w))
    return out


def sd_global_l2(sd) -> float:
    return float(torch.sqrt(sum((v.to(torch.float32) ** 2).sum()
                                for v in sd.values())))


def sd_clip_(sd, bound: float):
    n = sd_global_l2(sd)
    s = min(1.0, bound / (n + 1e-6))
    for k in sd:
        sd[k] = sd[k] * s
    return sd


def sd_add_noise_(sd, sigma: float, gen: torch.Generator):
    for k in sd:
        sd[k] = sd[k] + torch.normal(0.0, sigma, size=sd[k].shape,
                                     generator=gen, dtype=torch.float32)
    return sd


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------

class Client:
    def __init__(self, cid: int, model: nn.Module, loader, cfg, device,
                 seed: int) -> None:
        self.id = cid
        self.model = copy.deepcopy(model).to(device)
        self.loader = loader
        self.n = len(loader.dataset)
        self.cfg = cfg
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def sync(self, sd) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in sd.items()})

    def local_update(self, epochs: int):
        """Returns (payload_sd, is_delta, mean_loss)."""
        cfg, dev = self.cfg, self.device
        opt = torch.optim.SGD(self.model.parameters(), lr=cfg.lr,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
        before = {k: v.detach().cpu().clone()
                  for k, v in self.model.state_dict().items()}
        losses = []
        self.model.train()
        if cfg.dp_mode == "cg-ng":
            # Clip-Gradient / Noise-Gradient (LDP-style, per batch):
            # every batch gradient is L2-clipped then perturbed with
            # per-coordinate Gaussian noise before the optimizer step.
            params = [p for p in self.model.parameters() if p.requires_grad]
            for _ in range(epochs):
                for x, y in self.loader:
                    x, y = x.to(dev), y.to(dev)
                    opt.zero_grad()
                    loss = F.cross_entropy(self.model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip)
                    for p in params:
                        noise = torch.normal(0.0, cfg.sigma,
                                             size=p.grad.shape,
                                             generator=self.gen,
                                             dtype=torch.float32).to(dev)
                        p.grad.add_(noise)
                    opt.step()
                    losses.append(float(loss.item()))
        else:
            for _ in range(epochs):
                for x, y in self.loader:
                    x, y = x.to(dev), y.to(dev)
                    opt.zero_grad()
                    loss = F.cross_entropy(self.model(x), y)
                    loss.backward()
                    if cfg.dp_mode == "cg-np":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       cfg.clip)
                    opt.step()
                    losses.append(float(loss.item()))
        after = {k: v.detach().cpu().clone()
                 for k, v in self.model.state_dict().items()}
        mean_loss = float(np.mean(losses)) if losses else 0.0

        if cfg.dp_mode in ("baseline", "cg-ng"):
            # cg-ng noise was injected into the final-epoch gradient step.
            return after, False, mean_loss
        if cfg.dp_mode == "cg-np":
            sd_add_noise_(after, cfg.sigma, self.gen)
            return after, False, mean_loss
        if cfg.dp_mode == "cp-np":
            delta = {k: after[k].to(torch.float32) - before[k].to(torch.float32)
                     for k in after}
            sd_clip_(delta, cfg.clip)
            sd_add_noise_(delta, cfg.sigma, self.gen)
            return delta, True, mean_loss
        raise ValueError(f"unknown dp_mode {cfg.dp_mode}")


# -----------------------------------------------------------------------------
# Edge
# -----------------------------------------------------------------------------

class Edge:
    def __init__(self, eid: int, init_sd) -> None:
        self.id = eid
        self.sd = {k: v.cpu().clone() for k, v in init_sd.items()}
        self.round_samples: List[int] = []   # m_u per intermediate round
        self._buf: List[dict] = []
        self._buf_n: List[int] = []
        self._buf_delta = False

    def refresh(self) -> None:
        self._buf, self._buf_n = [], []

    def receive(self, payload, n: int, is_delta: bool) -> None:
        self._buf.append(payload)
        self._buf_n.append(n)
        self._buf_delta = is_delta

    def aggregate(self, edge_weighting: str, eta: float = 1.0) -> None:
        if not self._buf:
            return
        # drop non-finite payloads (diverged clients / extreme noise) instead
        # of letting one NaN corrupt the edge model
        keep = [i for i, sd in enumerate(self._buf)
                if all(torch.isfinite(v).all() for v in sd.values())]
        if len(keep) < len(self._buf):
            print(f"[edge {self.id}][WARNING] dropped "
                  f"{len(self._buf) - len(keep)}/{len(self._buf)} non-finite "
                  "client payload(s) — check lr / DP sigma calibration "
                  "(see scripts/calibrate_sigma.py)")
        if not keep:
            self.round_samples.append(0)
            self.refresh()
            return
        buf = [self._buf[i] for i in keep]
        buf_n = [self._buf_n[i] for i in keep]
        w = (buf_n if edge_weighting == "samplesize" else [1.0] * len(buf))
        agg = sd_weighted_average(buf, w)
        if self._buf_delta:
            self.sd = {k: self.sd[k].to(torch.float32) + eta * agg[k]
                       for k in self.sd}
        else:
            self.sd = agg
        self.round_samples.append(int(sum(buf_n)))
        self.refresh()

    def avg_samples(self) -> float:
        """n_j: average per-intermediate-round sample total (ESCS statistic)."""
        return float(np.mean(self.round_samples)) if self.round_samples else 1.0

    def reset_round_samples(self) -> None:
        self.round_samples = []


# -----------------------------------------------------------------------------
# Cloud
# -----------------------------------------------------------------------------

class Cloud:
    def __init__(self, init_sd, cfg, d_res: float) -> None:
        self.sd = {k: v.cpu().clone() for k, v in init_sd.items()}
        self.cfg = cfg
        # Divergence-aware selection is decided ONCE, from public aggregates
        # (post-processing; zero privacy cost).
        if cfg.central_weighting == "auto":
            self.rule = "avg_samplesize" if d_res <= cfg.tau else "accuracy"
        else:
            self.rule = cfg.central_weighting
        self.selected_rule = self.rule

    @torch.no_grad()
    def _accuracy(self, sd, model: nn.Module, val_loader, device) -> float:
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        model.eval()
        correct = total = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(y.numel())
        return correct / max(total, 1)

    def aggregate(self, edges: List[Edge], probe_model: nn.Module,
                  val_loader, device) -> Dict[str, float]:
        info: Dict[str, float] = {}
        sds = [e.sd for e in edges]
        if self.rule == "uniform":
            w = [1.0] * len(edges)
        elif self.rule == "avg_samplesize":                    # ESCS
            w = [e.avg_samples() for e in edges]
        elif self.rule == "accuracy":                          # ESCA
            accs = [self._accuracy(e.sd, probe_model, val_loader, device)
                    for e in edges]
            info["edge_acc_spread"] = float(np.std(accs))
            w = [max(a, 1e-6) for a in accs]
        else:
            raise ValueError(f"unknown central weighting {self.rule}")
        wn = np.asarray(w, dtype=np.float64)
        wn = wn / wn.sum()
        info["weight_entropy"] = float(-(wn * np.log(wn + 1e-12)).sum())
        self.sd = sd_weighted_average(sds, w)
        return info
