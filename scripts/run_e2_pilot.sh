#!/usr/bin/env bash
# Gate G1: H-DPBA vs oracle vs random vs nonprivate, 3 seeds.
# Pass --dataset mnist for real data.
set -e
for SEED in 42 123 2024; do
  for ASSIGN in random oracle hdpba nonprivate; do
    python -m hdpba_hfl.train --config configs/e2_pilot.yaml \
      --assignment $ASSIGN --name e2_${ASSIGN} --seed $SEED "$@"
  done
done
python -m hdpba_hfl.stats runs --metric final_acc --compare e2_hdpba e2_random
python -m hdpba_hfl.stats runs --metric final_acc --compare e2_hdpba e2_oracle
