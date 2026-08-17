#!/usr/bin/env bash
set -e
for SEED in 42 123 2024 7 99; do
  for ASSIGN in random hdpba; do
    for METHOD in bl1 bl2 escs esca auto; do
      python -m hdpba_hfl.train --config configs/e1_main.yaml \
        --assignment $ASSIGN --method $METHOD \
        --name e1_${ASSIGN}_${METHOD} --seed $SEED "$@"
    done
  done
done
