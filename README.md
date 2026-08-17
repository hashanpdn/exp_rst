# H-DPBA + Dual-Weight DP-HFL

Fresh reference implementation for:

> **Privacy-Preserving Model Aggregation for Hierarchical Federated Learning**
> H-DPBA (hierarchical DP balanced assignment) + ESCS/ESCA weighting +
> divergence-aware selection, with end-to-end RDP accounting.

This is a clean-room implementation. It is *interface-compatible in spirit*
with [`frontloaded-dp-hfl`](https://github.com/hashanpdn/frontloaded-dp-hfl)
(same DP client modes `baseline / cg-ng / cg-np / cp-np`, same topology
semantics `T / U(tau_2) / E(tau_1)`, `cfrac -> client_frac`,
`efrac -> edge_frac`), but shares no code with it.

---

## What is implemented

| Component | Where | Paper element |
|---|---|---|
| H-DPBA private client-to-edge assignment (exp-mech pull protocol, R passes, capacity + feasibility, strict-cap rebalance) | `hdpba_hfl/assignment.py` | Algorithm 1 |
| Divergence-aware selector (ESCS vs ESCA from public D_res, zero budget) | `hdpba_hfl/fl_core.py` (`Cloud`) | Algorithm 2 |
| Hierarchical training loop with edge/central weighting and DP uploads | `hdpba_hfl/train.py`, `fl_core.py` | Algorithm 3 |
| ESCS / ESCA / BL1 / BL2 / auto presets | `hdpba_hfl/options.py` | §Method |
| DP primitives: Gaussian calibration, exponential mechanism (Gumbel trick), clipping, simplex projection | `hdpba_hfl/privacy.py` | §Privacy |
| **RDP accountant**: Gaussian, sampled-Gaussian (Mironov'19 integer-α), pure-DP events, hybrid conversion | `hdpba_hfl/privacy.py` | Theorem 2 |
| Assignment baselines: random (A0), oracle (A1), non-private greedy (A2), RR binary-presence (A3, pp-CFL-style), H-DPBA (A4) | `hdpba_hfl/assignment.py` | Eval axis A |
| Partitions: IID, pathological k-class (P1), Dirichlet (P2), power-law (P3), compound (P4) | `hdpba_hfl/datasets.py` | Eval §2.1 |
| Models: LeNet (MNIST), Tanh-LeNet (FMNIST), simplified AlexNet (CIFAR-10) | `hdpba_hfl/models.py` | §Setup |
| Offline synthetic dataset (Gaussian class blobs) for CI / no-internet dev | `hdpba_hfl/datasets.py` | — |
| Sweep runner (resumable), curve plots, seed aggregation + Wilcoxon + Cliff's δ | `scripts/`, `hdpba_hfl/stats.py` | Eval §5 |
| 32 unit + end-to-end tests | `tests/` | — |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                      # 32 tests, ~30 s CPU

# 30-second proof of the core mechanism (no training needed):
python scripts/demo_assignment.py
#   recovery of oracle gain: 0.22 @ eps_a=0.1 -> 0.87 @ 1.0 (monotone),
#   RR prior-art baseline stuck at ~0.14.

# One full run (synthetic offline data):
python -m hdpba_hfl.train --config configs/e2_pilot.yaml

# Same on real MNIST (downloads via torchvision):
python -m hdpba_hfl.train --config configs/e2_pilot.yaml --dataset mnist

# An experiment from the evaluation framework (resumable; re-run to resume):
python scripts/sweep.py sweeps/e1_main.yaml --outdir runs/e1
python -m hdpba_hfl.stats runs/e1 --metric final_acc \
    --compare method-auto_assignment-hdpba method-esca_assignment-random
python scripts/plot_curves.py runs/e1 --out fig_e1.png
```

## Repository layout

```
hdpba_hfl/
  privacy.py      DP primitives + RDP accountant
  datasets.py     loaders (mnist|fmnist|cifar10|synthetic) + partitions P1-P4
  models.py       LeNetMNIST, TanhLeNet, AlexNetS
  assignment.py   A0-A4 assignment methods + diagnostics
  fl_core.py      Client (4 DP modes), Edge, Cloud (4 rules + auto)
  train.py        end-to-end loop, CSV/JSON logging, accounting
  options.py      dataclass Config <- YAML <- CLI; method presets
  stats.py        seed aggregation, Wilcoxon, Cliff's delta
configs/          single-run YAMLs (E1 cell, E2 pilot, E6 dropout)
sweeps/           grid specs mirroring evaluation-framework experiments E1-E8
scripts/          sweep.py, plot_curves.py, demo_assignment.py, run_*.sh
tests/            test_core.py (mechanism-level), test_smoke.py (end-to-end)
```

## Key CLI flags (all Config fields are flags)

```
--method {bl1,bl2,escs,esca,auto}      aggregation preset
--assignment {random,oracle,nonprivate,rr,hdpba}
--dp_mode {baseline,cg-ng,cg-np,cp-np} --clip C --sigma S
--dataset {synthetic,mnist,fmnist,cifar10}
--partition {iid,pathological,dirichlet,powerlaw,compound}
--num_clients K --num_edges J --client_frac q --edge_frac f
--global_rounds T --intermediate_rounds U --local_epochs E
--eps_assign eps_a --eps_agg eps_g --assign_passes R --tau 0.15
```

## Design notes (read before extending)

1. **Privacy surface is closed.** Servers only ever observe: (a) Gaussian-
   noised secure aggregates, (b) exponential-mechanism choices, (c) clipped +
   noised model uploads. Everything else (selector, ESCA accuracies on the
   *public* server-side validation split, rebalancing) is post-processing.
   The accountant composes exactly these events; `summary.json` lists them.
2. **Privacy positioning (deliberate).** The *formal* DP guarantee attaches
   to the assignment mechanism (exponential-mechanism choices + Gaussian
   aggregate releases, RDP-composed -> `eps_total`). Training perturbation
   defaults to **CG-NG small-noise LDP-style** (per-batch clip-gradient +
   noise-gradient; verified utility-viable at clip=5.0, sigma=0.02,
   lr=0.05), matching the published preliminary study's philosophy: the
   contribution is heterogeneity management under realistic perturbation.
   CG-modes are **not** composed into eps_total (a fake epsilon is worse
   than none); they are itemized in the ledger as
   `training_perturbation (... LDP-style, NOT composed)`. For a formal
   end-to-end training budget use `cp-np` — but note strict model-level
   accounting at small eps is utility-infeasible in this regime
   (run `scripts/calibrate_sigma.py` for the numbers).
3. **Balanced budget split matters.** ε_a (choice) and ε_g (aggregate
   release) must scale together: confident choices computed on very noisy
   aggregates herd onto noise (empirically visible — recovery becomes
   non-monotone in ε_a if ε_g is held small). Default pairing ε_g = ε_a/2.
4. **Capacity deadlock + rebalance.** A perfectly balanced start blocks all
   single moves; passes therefore run with slack cap+1 and finish with a
   *data-independent* random rebalance to strict cap (costs no budget).
5. **`refresh_per_batch=True`** (clients see updated local copies of the
   aggregates during a pass) avoids herding; setting it False saves nothing
   in this simulation and degrades quality — kept only for ablation.
6. **A1 ≡ A2 in this implementation** (oracle and non-private greedy share
   the optimizer at ε = ∞): the interesting gap is privacy noise, not
   optimizer quality; a stronger combinatorial oracle can be added if a
   reviewer asks.
7. **`d_res_true` / `phi_true` are simulation-only diagnostics** computed
   from ground-truth histograms; the deployed protocol never has them.
   Public `d_res` / `phi` are computed from published noisy aggregates and
   are what the selector uses.
8. **Determinism.** Every stochastic component draws from a seeded generator
   (`--seed`); dataset partitions, assignment, participation sampling, and
   DP noise are all reproducible. GPU nondeterminism caveats aside, CPU runs
   are bit-reproducible.

## Mapping to the evaluation framework

| Experiment | Sweep spec | Notes |
|---|---|---|
| E1 main comparison | `sweeps/e1_main.yaml` | edit `fixed:` per dataset/partition |
| E2 assignment quality (Gate G1) | `sweeps/e2_assignment_quality.yaml` + `scripts/demo_assignment.py` | demo = seconds; sweep = with training |
| E2b ε_a × R sensitivity | `sweeps/e2b_epsa_sweep.yaml` | |
| E3 skew degree | `sweeps/e3_skew_degree.yaml` | |
| E4 component ablation | `sweeps/e4_ablation.yaml` | uses explicit weighting fields |
| E6 edge dropout | `sweeps/e6_edge_dropout.yaml` | headline robustness figure |
| E7 (E,U) allocation | `sweeps/e7_eu_allocation.yaml` | run 4× with the (E,U) pairs |
| E8b budget split | `sweeps/e8_budget_split.yaml` | |
| E8c empirical attacks | not included | MIA harness is deliberately separate; add under `attacks/` when running E8c |

## Requirements

torch, torchvision, numpy, pandas, scipy, matplotlib, pyyaml, pytest
(see `requirements.txt`; CPU-only works for MNIST/FMNIST/synthetic).

## License / provenance

Fresh implementation written for this project; no code copied from
HierFL or frontloaded-dp-hfl. Conventions were aligned with the latter's
CLI/README for easy migration of experiment scripts.
