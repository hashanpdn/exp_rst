# Complete Command Set — CG-NG Small-Noise Direction
## (Heterogeneity is the headline; formal DP attaches to the assignment)

All commands for **Anaconda Prompt (Windows)**, run from the repo root
`D:\Courses\hdpba-hfl`. Re-running any sweep resumes it (finished runs skip).

---

## 0. After replacing the code with the new zip

```bat
conda activate hdpba
cd D:\Courses\hdpba-hfl
python -m pytest tests\ -q
:: expect: 32 passed
```

## 1. Sanity: verified-working pilot (synthetic, ~10 min CPU)

```bat
python -m hdpba_hfl.train --config configs\e2_pilot.yaml
:: expect: learning curve rising well above 10% chance; eps_total ~ 0.7
:: (eps_total = ASSIGNMENT budget; cg-ng noise is ledgered, not composed)
```

## 2. GATE G1 — the decisive pilot on real MNIST (~1-2 h CPU, minutes GPU)

```bat
python -m hdpba_hfl.train --config configs\e2_pilot.yaml --dataset mnist
python -m hdpba_hfl.train --config configs\e2_pilot.yaml --dataset mnist --assignment oracle
python -m hdpba_hfl.train --config configs\e2_pilot.yaml --dataset mnist --assignment random
:: PASS: hdpba final_acc within ~2 points of oracle, both clearly above random
```

If lr=0.05 misbehaves on MNIST (oscillation): retry `--lr 0.01`; record the
choice — it goes in the hyperparameter appendix.

## 3. Full E2 (all assignment methods x 3 seeds)

```bat
python scripts\sweep.py sweeps\e2_assignment_quality.yaml --outdir runs\e2
python -m hdpba_hfl.stats runs\e2 --metric final_acc
python -m hdpba_hfl.stats runs\e2 --metric final_acc --compare assignment-hdpba assignment-oracle
python -m hdpba_hfl.stats runs\e2 --metric final_acc --compare assignment-hdpba assignment-random
python scripts\sweep.py sweeps\e2b_epsa_sweep.yaml --outdir runs\e2b
```

## 4. E1 — main heterogeneity comparison (Table 1 / Fig 1)

```bat
:: MNIST, pathological k=2 (as configured):
python scripts\sweep.py sweeps\e1_main.yaml --outdir runs\e1_mnist_p1

:: then edit sweeps\e1_main.yaml "fixed:" block and repeat for:
::   dataset: fmnist | cifar10
::   partition: dirichlet (alpha 0.5) | powerlaw | compound
python scripts\sweep.py sweeps\e1_main.yaml --outdir runs\e1_fmnist_p1

:: tables + curves:
python -m hdpba_hfl.stats runs\e1_mnist_p1 --metric final_acc
python -m hdpba_hfl.stats runs\e1_mnist_p1 --metric acc_auc
python -m hdpba_hfl.stats runs\e1_mnist_p1 --metric final_worst_class_recall
python scripts\plot_curves.py runs\e1_mnist_p1 --out fig1_mnist.png
```

## 5. E3 skew degree · E4 ablation · E6 dropout (headline robustness)

```bat
python scripts\sweep.py sweeps\e3_skew_degree.yaml --outdir runs\e3
python scripts\sweep.py sweeps\e4_ablation.yaml   --outdir runs\e4
python scripts\sweep.py sweeps\e6_edge_dropout.yaml --outdir runs\e6
python -m hdpba_hfl.stats runs\e6 --metric acc_auc
python scripts\plot_curves.py runs\e6 --out fig4_dropout.png --metric test_acc
```

## 6. E7 — (E,U) allocation

```bat
for %%P in ("6 2" "4 3" "3 4" "2 6") do (for /f "tokens=1,2" %%a in (%%P) do ^
python -m hdpba_hfl.train --config configs\e1_main.yaml --local_epochs %%a --intermediate_rounds %%b --outdir runs\e7\e%%au%%b --name e%%au%%b)
python -m hdpba_hfl.stats runs\e7 --metric final_acc
```

(Or run the four `python -m hdpba_hfl.train ... --local_epochs E --intermediate_rounds U` lines individually.)

## 7. E8 — noise sensitivity (revised for CG-NG) + formal assignment budget

```bat
:: does the heterogeneity benefit survive increasing LDP noise?
python scripts\sweep.py sweeps\e8_noise_sensitivity.yaml --outdir runs\e8
python -m hdpba_hfl.stats runs\e8 --metric final_acc

:: the FORMAL privacy axis: assignment budget (pair eps_agg = eps_assign/2)
python -m hdpba_hfl.train --config configs\e2_pilot.yaml --dataset mnist --eps_assign 0.1 --eps_agg 0.05 --outdir runs\e8b\ea01 --name ea01
python -m hdpba_hfl.train --config configs\e2_pilot.yaml --dataset mnist --eps_assign 0.2 --eps_agg 0.1  --outdir runs\e8b\ea02 --name ea02
python -m hdpba_hfl.train --config configs\e2_pilot.yaml --dataset mnist --eps_assign 0.5 --eps_agg 0.25 --outdir runs\e8b\ea05 --name ea05
python -m hdpba_hfl.train --config configs\e2_pilot.yaml --dataset mnist --eps_assign 1.0 --eps_agg 0.5  --outdir runs\e8b\ea10 --name ea10
python -m hdpba_hfl.stats runs\e8b --metric final_acc
```

## 8. Assignment-only demo (paper number, 30 s)

```bat
python scripts\demo_assignment.py --seeds 42 123 2024 7 99
```

## 9. Optional: formal-DP reference point (appendix honesty note)

```bat
python scripts\calibrate_sigma.py --eps 5.0 --model_dim 60000
:: shows why strict model-level accounting at small eps is utility-infeasible
:: here -> justifies the LDP-style choice in one appendix paragraph
```

---

## How results are reported (unchanged from the measurement protocol)

- Every number: mean ± std over seeds (5 for E1/E2/E6, 3 elsewhere).
- eps_total in tables = **assignment budget (formal)**; training noise
  reported as "CG-NG, clip=5.0, sigma=..., LDP-style" in the setup section.
- `summary.json: training_dp_style` field distinguishes
  formal / ldp_unaccounted / none per run — use it when building tables.
