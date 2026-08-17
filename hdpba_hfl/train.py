"""Main entry point: assignment -> three-tier DP-HFL training -> logs + accounting.

Usage:
    python -m hdpba_hfl.train --config configs/e2_pilot.yaml \
        --assignment hdpba --method esca --seed 42
Outputs (under {outdir}/{name}_seed{seed}/):
    metrics.csv    per-global-round metrics
    summary.json   final metrics, D_res, selected rule, eps_total, config
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import random
import time

import numpy as np
import torch

from .assignment import build_assignment
from .datasets import build_loaders, client_histograms, load_dataset, make_partitions
from .fl_core import Client, Cloud, Edge
from .models import build_model
from .options import load_config
from .privacy import RDPAccountant


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(sd, model, loader, device, num_classes: int = 10):
    """Returns (accuracy, per_class_recall ndarray)."""
    model.load_state_dict({k: v.to(device) for k, v in sd.items()})
    model.eval()
    correct = total = 0
    cls_correct = np.zeros(num_classes, dtype=np.int64)
    cls_total = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            hit = (pred == y)
            correct += int(hit.sum())
            total += int(y.numel())
            for c in range(num_classes):
                mask = (y == c)
                cls_total[c] += int(mask.sum())
                cls_correct[c] += int((hit & mask).sum())
    recall = cls_correct / np.maximum(cls_total, 1)
    return correct / max(total, 1), recall


def main(argv=None) -> dict:
    cfg = load_config(argv)
    set_seed(cfg.seed)
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else (cfg.device if cfg.device != "auto" else "cpu"))
    rng = np.random.default_rng(cfg.seed)

    run_dir = os.path.join(cfg.outdir, f"{cfg.name}_seed{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- data ----------------------------------------------------------------
    train_set, test_set, ncls, shape = load_dataset(cfg.dataset, cfg.data_root,
                                                    cfg.seed)
    parts = make_partitions(cfg, train_set, ncls, rng)
    hist = client_histograms(train_set, parts, ncls)  # LOCAL knowledge only
    loaders, val_loader, test_loader = build_loaders(
        train_set, test_set, parts, cfg.batch_size, cfg.val_fraction, rng)

    # ---- privacy accountant + assignment --------------------------------------
    accountant = RDPAccountant()
    assign_res = build_assignment(cfg, hist, rng, accountant)
    print(f"[assignment] method={assign_res.method}  D_res={assign_res.d_res:.4f}"
          f"  Phi={assign_res.phi:.4f}  (true: D_res={assign_res.d_res_true:.4f}"
          f" Phi={assign_res.phi_true:.4f})")

    # ---- models / actors -------------------------------------------------------
    model = build_model(cfg.dataset, shape, ncls).to(device)
    probe = build_model(cfg.dataset, shape, ncls).to(device)
    init_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    clients = [Client(k, model, loaders[k], cfg, device, cfg.seed * 1009 + k)
               for k in range(cfg.num_clients)]
    edges = [Edge(j, init_sd) for j in range(cfg.num_edges)]
    edge_clients = {j: [k for k in range(cfg.num_clients)
                        if assign_res.assignment[k] == j]
                    for j in range(cfg.num_edges)}
    cloud = Cloud(init_sd, cfg, assign_res.d_res)
    print(f"[cloud] central rule selected: {cloud.selected_rule}"
          + (f" (auto, tau={cfg.tau})" if cfg.central_weighting == "auto" else ""))

    # ---- training-DP accounting -------------------------------------------------
    # cp-np: rigorous per-upload guarantee (released delta = clipped + noised;
    #        sensitivity = clip exactly) -> composed into eps_total.
    # cg-ng / cg-np: LDP-style perturbation whose release also depends on
    #        un-noised intermediate state -> NOT composed into eps_total
    #        (a fake number is worse than none); recorded in the ledger and
    #        reported as "training: LDP-style, sigma=..., unaccounted".
    if cfg.dp_mode == "cp-np":
        uploads = cfg.global_rounds * cfg.intermediate_rounds
        accountant.add_sampled_gaussian(q=cfg.client_frac * cfg.edge_frac,
                                        sigma=cfg.sigma, sensitivity=cfg.clip,
                                        count=uploads, label="training_upload")
    elif cfg.dp_mode != "baseline":
        accountant.events.append(
            f"training_perturbation ({cfg.dp_mode}, sigma={cfg.sigma}, "
            f"clip={cfg.clip}) [LDP-style, NOT composed into eps_total]")
        print(f"[privacy][NOTE] dp_mode={cfg.dp_mode}: training noise is "
              "LDP-style and NOT composed into eps_total; eps_total covers "
              "the assignment mechanism only. Use cp-np for a formal "
              "end-to-end training budget.")

    # ---- main loop ----------------------------------------------------------------
    csv_path = os.path.join(run_dir, "metrics.csv")
    fields = ["round", "test_acc", "worst_class_recall", "class_recall_std",
              "train_loss", "edge_acc_spread", "weight_entropy", "seconds"]
    history = []
    t0 = time.time()
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for t in range(1, cfg.global_rounds + 1):
            n_sel_edges = max(1, int(round(cfg.edge_frac * cfg.num_edges)))
            sel_edges = rng.choice(cfg.num_edges, n_sel_edges, replace=False)
            round_losses = []
            for j in sel_edges:
                edges[j].sd = {k: v.clone() for k, v in cloud.sd.items()}
                edges[j].reset_round_samples()
            for _u in range(cfg.intermediate_rounds):
                for j in sel_edges:
                    pool = edge_clients[j]
                    if not pool:
                        continue
                    n_sel = max(1, int(round(cfg.client_frac * len(pool))))
                    sel = rng.choice(pool, n_sel, replace=False)
                    edges[j].refresh()
                    for k in sel:
                        clients[k].sync(edges[j].sd)
                        payload, is_delta, loss = clients[k].local_update(
                            cfg.local_epochs)
                        edges[j].receive(payload, clients[k].n, is_delta)
                        round_losses.append(loss)
                    edges[j].aggregate(cfg.edge_weighting, cfg.eta)
            info = cloud.aggregate([edges[j] for j in sel_edges], probe,
                                   val_loader, device)
            acc, recall = evaluate(cloud.sd, probe, test_loader, device)
            row = {
                "round": t,
                "test_acc": round(acc, 5),
                "worst_class_recall": round(float(recall.min()), 5),
                "class_recall_std": round(float(recall.std()), 5),
                "train_loss": round(float(np.mean(round_losses)), 5)
                if round_losses else None,
                "edge_acc_spread": round(info.get("edge_acc_spread", float("nan")), 5),
                "weight_entropy": round(info.get("weight_entropy", float("nan")), 5),
                "seconds": round(time.time() - t0, 1),
            }
            history.append(row)
            writer.writerow(row)
            fcsv.flush()
            if t % cfg.log_every == 0:
                print(f"[round {t:3d}/{cfg.global_rounds}] "
                      f"acc={acc:.4f} loss={row['train_loss']}")

    # ---- summary --------------------------------------------------------------
    eps_total = accountant.get_epsilon(cfg.delta)
    accs = [h["test_acc"] for h in history]
    summary = {
        "config": dataclasses.asdict(cfg),
        "assignment_method": assign_res.method,
        "d_res": assign_res.d_res,
        "phi": assign_res.phi,
        "d_res_true": assign_res.d_res_true,
        "phi_true": assign_res.phi_true,
        "selected_central_rule": cloud.selected_rule,
        "final_acc": accs[-1],
        "best_acc": max(accs),
        "acc_auc": float(np.mean(accs)),
        "final_worst_class_recall": history[-1]["worst_class_recall"],
        "final_class_recall_std": history[-1]["class_recall_std"],
        "eps_total": eps_total,
        "eps_is_formal": cfg.dp_mode in ("baseline", "cp-np"),
        "training_dp_style": ("formal" if cfg.dp_mode == "cp-np" else
                              "none" if cfg.dp_mode == "baseline" else "ldp_unaccounted"),
        "delta": cfg.delta,
        "privacy_events": accountant.events,
        "wallclock_sec": time.time() - t0,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[privacy]\n" + accountant.summary(cfg.delta))
    print(f"[done] final={summary['final_acc']:.4f} best={summary['best_acc']:.4f}"
          f" eps_total={eps_total:.3f}  -> {run_dir}")
    return summary


if __name__ == "__main__":
    main()
