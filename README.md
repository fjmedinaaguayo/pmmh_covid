# SEIR PMMH for COVID-19 in Mexico: paper code

Code to reproduce the figures of the paper. The model is a weekly SEIR with a
Brownian-motion log transmission rate, fitted by PMMH with a negative-binomial
observation model whose mean is `modeled infections / eta` (`eta` = underreporting factor).

## Layout

| File | Purpose |
|---|---|
| `pmmh_seir.py` | PMMH sampler (writes `<chain>.npz` and `<chain>_metadata.json`) |
| `plot_paper_figures.py` | R_t, beta_t, weekly cases, susceptibles, posteriors (kappa, gamma, sigma, nu), traces, ACFs, correlation matrix |
| `nowcast_sequential.py` | Observed-case nowcasts from truncated chains |
| `run_all.sh` | Regenerates all figures |
| `data/` | Weekly COVID-19 cases and INEGI 2020 state populations |
| `chains/` | PMMH output (not tracked; see `.gitignore`) |

## Reproducing the chains

```bash
pip install -r requirements.txt

# Full data, different underreporting factors
for eta in 1 7 15; do
  python pmmh_seir.py --eta $eta --save-file chains/pmmh_eta${eta}.npz
done

# Truncated fits (eta = 15) used for nowcasts
for w in 20 60 75 85; do
  python pmmh_seir.py --eta 15 --max-weeks $w --save-file chains/pmmh_truncated_w${w}.npz
done
```

## Reproducing the figures

```bash
./run_all.sh
```

Figures go to `figures/`, with an `_eta{1,7,15}` suffix for the full-data chains
and `posterior_observed_nowcast_sequential_w{20,60,75,85}.png` for the nowcasts.
