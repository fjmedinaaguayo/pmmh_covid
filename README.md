# COVID-19 SEIR Model Analysis

This repository contains implementations of Sequential Monte Carlo (SMC) and Particle Marginal Metropolis-Hastings (PMMH) methods for COVID-19 epidemiological modeling using SEIR (Susceptible-Exposed-Infectious-Recovered) dynamics.

## Files

### Core Implementation
- `pmmh_seir.py` - Main PMMH implementation for SEIR model with Brownian motion on transmission rate
- `seir_smc_final.py` - Sequential Monte Carlo implementation for SEIR model
- `generic_smc.py` - Generic SMC framework

### Analysis
- `Revised Functions.ipynb` - Jupyter notebook with analysis and function revisions

### Data Files
- `covid_df.feather` - COVID-19 case data
- `INEGI_2020_State_Population.feather` - Population data from INEGI 2020
- `pmmh_chain.npz` - Saved MCMC chain results

## Features

### PMMH Implementation (`pmmh_seir.py`)
- Particle Marginal Metropolis-Hastings for Bayesian inference
- Adaptive Metropolis with empirical covariance learning
- Automatic checkpoint saving
- Joint sampling of parameters and latent trajectories

### SMC Implementation (`seir_smc_final.py`)
- Sequential Monte Carlo for state estimation
- Particle filtering for SEIR dynamics
- Systematic resampling
- Effective sample size monitoring

## Model Description

The SEIR model includes:
- **S(t)**: Susceptible population
- **E(t)**: Exposed population
- **I(t)**: Infectious population
- **R(t)**: Recovered population
- **β(t)**: Time-varying transmission rate (follows Brownian motion)

## Usage

```python
# Run PMMH analysis
python pmmh_seir.py

# Run SMC analysis
python seir_smc_final.py
```

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
- jupyter (for notebook analysis)

## Authors

Research collaboration on COVID-19 epidemiological modeling.