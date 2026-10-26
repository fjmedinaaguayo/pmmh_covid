#!/usr/bin/env bash
# Re-create every figure of the paper from the chains in ./chains.
set -euo pipefail

for eta in 1 7 15; do
  python plot_paper_figures.py --chain "chains/pmmh_eta${eta}.npz" --tag "eta${eta}"
done

python nowcast_sequential.py
