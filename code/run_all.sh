#!/bin/bash
# Reproduces every number, table and figure of the copula paper (about 1 h on 2 cores).
#   1. main experiment (730 cases x 3 lead levels, with placebo transplants)
#   2. sensitivity sweep (16 configurations, 300 cases, medium lead)
#   3. bivariate finite-ensemble check of the transplant
#   4. figures, LaTeX tables and the auto-generated results text
set -e
cd "$(dirname "$0")"
mkdir -p ../results
export OPENBLAS_NUM_THREADS=2
python3 simulate_wind.py --out ../results/main.npz --n 730 --rhos 0.8 0.5 0.2 \
  --sigma_beta 0.2 --kappa 0.85 --ellA 150 > ../results/main.log 2>&1
for tau in 0.0 0.1 0.2 0.3; do
  for ellA in 100 150 250 400; do
    python3 simulate_wind.py --out ../results/sweep_tau${tau}_ellA${ellA}.npz --n 300 --rhos 0.5 \
      --sigma_beta $tau --kappa 0.85 --ellA $ellA --light --no_placebo --seed 777 \
      > ../results/sweep_tau${tau}_ellA${ellA}.log 2>&1
  done
done
python3 leakage_check.py > ../results/leakage_check.log 2>&1
python3 make_figures.py
python3 make_tables.py > /dev/null
python3 make_results_text.py > /dev/null
echo ALLDONE
