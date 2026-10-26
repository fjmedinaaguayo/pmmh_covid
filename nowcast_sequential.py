"""
Sequential nowcasting from a saved PMMH chain using bootstrap updates.

This script differs from open-loop forecasting by conditioning on each newly
observed y_{t+h} as it arrives. For a fixed start index t:
1) propagate particles from t to t+1,
2) weight by p(y_{t+1} | state_{t+1}, theta),
3) resample,
4) repeat for t+2, ...

At each step, the script also stores the one-step-ahead predictive distribution
for observed weekly cases, so outputs and plots are directly comparable to the
original forecast script.

Important modeling choice:
- Static parameters theta = (kappa, gamma, sigma, nu) are sampled once per
  particle from the PMMH posterior and then kept fixed along that lineage.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.special import logsumexp


COVID_DATA_FILE = "./data/covid_df.feather"
WEEKLY_GROUP_FREQ = "W-SUN"
BURN_IN_FRACTION = 0.2
NUM_PARTICLES = 5000
RANDOM_SEED = 42
HORIZON_WEEKS = 52
RESAMPLE_ESS_RATIO = 1.0
TRAINING_WEEKS = [20, 60, 75, 85]


def _load_chain(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    required = ["samples", "states_paths", "Y_obs"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Chain file is missing required keys: {missing}")

    out = {
        "samples": data["samples"],
        "states_paths": data["states_paths"],
        "Y_obs": data["Y_obs"],
        "prior_spec": None,
    }

    if "prior_spec_json" in data:
        try:
            out["prior_spec"] = json.loads(data["prior_spec_json"].item())
        except Exception:
            out["prior_spec"] = None

    return out


def _load_underreporting_factor(chain_file):
    metadata_file = os.path.splitext(chain_file)[0] + "_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        try:
            return float(
                metadata["model_specification"]["observation_model"][
                    "underreporting_factor"
                ]
            )
        except (KeyError, TypeError, ValueError):
            pass

    raise ValueError(f"Could not read underreporting_factor from '{metadata_file}'.")


def _load_weekly_dates(covid_data_file, weekly_group_freq):
    data = pd.read_feather(covid_data_file)
    weekly = (
        data.groupby([pd.Grouper(key="date", freq=weekly_group_freq)])
        .agg({"confirmed_cases": "sum"})
        .reset_index()
    )
    return weekly["date"].values, weekly["confirmed_cases"].values.astype(float)


def _systematic_resample(weights, rng):
    n = len(weights)
    positions = (np.arange(n) + rng.uniform()) / n
    return np.searchsorted(np.cumsum(weights), positions)


def _propagate_particles(states_t, params_particles, rng):
    """One-step SEIR propagation for all particles."""
    kappa = params_particles[:, 0]
    gamma = params_particles[:, 1]
    sigma = params_particles[:, 2]

    s_t = states_t[:, 0]
    e_t = states_t[:, 1]
    i_t = states_t[:, 2]
    r_t = states_t[:, 3]
    log_beta_t = states_t[:, 4]

    n_pop = s_t + e_t + i_t + r_t
    latent_inf = np.exp(log_beta_t) * s_t * i_t / n_pop
    obs_inf = kappa * e_t
    recov = gamma * i_t

    s_new = s_t - latent_inf
    e_new = e_t + latent_inf - obs_inf
    i_new = i_t + obs_inf - recov
    r_new = r_t + recov
    log_beta_new = log_beta_t + sigma * rng.normal(0.0, 1.0, size=len(s_t))

    states_next = np.column_stack([s_new, e_new, i_new, r_new, log_beta_new])
    return states_next, obs_inf


def _loglik_observation(y_t, obs_inf, nu, underreporting_factor):
    """Negative-binomial log-likelihood for observed weekly cases."""
    loglik = np.full_like(obs_inf, -np.inf, dtype=float)

    valid = (
        np.isfinite(obs_inf)
        & (obs_inf > 0.0)
        & np.isfinite(nu)
        & (nu > 0.0)
    )
    if np.any(valid):
        mean_observed = obs_inf[valid] / underreporting_factor
        p = nu[valid] / (nu[valid] + mean_observed)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        loglik[valid] = stats.nbinom(n=nu[valid], p=p).logpmf(y_t)

    return loglik


def _sequential_beta_nowcast(
    samples,
    states_paths,
    y_obs,
    start_idx,
    horizon_weeks,
    burn_in,
    num_particles,
    seed,
    underreporting_factor,
    resample_ess_ratio,
):
    n_mcmc, n_params = samples.shape
    if n_params < 4:
        raise ValueError(f"Expected parameter dimension >= 4, got {n_params}.")
    if burn_in >= n_mcmc:
        raise ValueError(f"burn_in={burn_in} is too large for chain length {n_mcmc}.")

    max_available = len(y_obs) - 1 - start_idx
    if max_available <= 0:
        raise ValueError(
            "No observations available after forecast_start_idx for sequential updating."
        )
    update_horizon = min(int(horizon_weeks), int(max_available))

    rng = np.random.default_rng(seed)
    posterior_idx = np.arange(burn_in, n_mcmc)
    chosen_idx = rng.choice(posterior_idx, size=num_particles, replace=True)

    params_particles = samples[chosen_idx, :4].astype(float).copy()
    states_t = states_paths[chosen_idx, :, start_idx].astype(float).copy()
    weights = np.ones(num_particles, dtype=float) / num_particles

    beta_particles = np.full((num_particles, update_horizon), np.nan, dtype=float)
    log_beta_particles = np.full((num_particles, update_horizon), np.nan, dtype=float)
    latent_pred_particles = np.full((num_particles, update_horizon), np.nan, dtype=float)
    observed_pred_particles = np.full((num_particles, update_horizon), np.nan, dtype=float)
    ess_history = np.full(update_horizon, np.nan, dtype=float)
    resampled_history = np.zeros(update_horizon, dtype=bool)

    for h in range(update_horizon):
        obs_index = start_idx + 1 + h
        y_new = y_obs[obs_index]

        states_pred, obs_inf = _propagate_particles(states_t, params_particles, rng)

        # One-step-ahead predictive observed cases (before seeing y_new).
        nu = params_particles[:, 3]
        valid_pred_obs = np.isfinite(obs_inf) & (obs_inf > 0.0) & np.isfinite(nu) & (nu > 0.0)
        sampled_observed = np.full(num_particles, np.nan, dtype=float)
        if np.any(valid_pred_obs):
            mean_observed = obs_inf[valid_pred_obs] / underreporting_factor
            p = nu[valid_pred_obs] / (nu[valid_pred_obs] + mean_observed)
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
            sampled_observed[valid_pred_obs] = stats.nbinom(
                n=nu[valid_pred_obs], p=p
            ).rvs(random_state=rng)

        latent_pred_particles[:, h] = obs_inf
        observed_pred_particles[:, h] = sampled_observed

        # Enforce biological constraints before weighting.
        valid_state = np.all(states_pred[:, :4] >= 0.0, axis=1) & np.isfinite(obs_inf)
        loglik = _loglik_observation(
            y_t=y_new,
            obs_inf=obs_inf,
            nu=params_particles[:, 3],
            underreporting_factor=underreporting_factor,
        )
        loglik[~valid_state] = -np.inf

        if not np.any(np.isfinite(loglik)):
            raise RuntimeError(
                f"All particles invalid at step h={h+1} (obs index {obs_index})."
            )

        logw = np.log(weights) + loglik
        logw -= logsumexp(logw)
        weights = np.exp(logw)

        ess = 1.0 / np.sum(weights**2)
        ess_history[h] = ess

        do_resample = ess < (resample_ess_ratio * num_particles)
        if do_resample:
            idx = _systematic_resample(weights, rng)
            states_pred = states_pred[idx]
            params_particles = params_particles[idx]
            weights = np.ones(num_particles, dtype=float) / num_particles
            resampled_history[h] = True

        states_t = states_pred

        log_beta_h = states_t[:, 4]
        beta_particles[:, h] = np.exp(log_beta_h)
        log_beta_particles[:, h] = log_beta_h

    return {
        "chosen_chain_indices": chosen_idx,
        "params_particles": params_particles,
        "beta_particles": beta_particles,
        "log_beta_particles": log_beta_particles,
        "latent_pred_particles": latent_pred_particles,
        "observed_pred_particles": observed_pred_particles,
        "ess_history": ess_history,
        "resampled_history": resampled_history,
        "update_horizon": update_horizon,
    }


def _quantile_summary(particles, probs=(2.5, 25.0, 50.0, 75.0, 97.5)):
    return {
        f"q{str(p).replace('.0', '').replace('.', '_')}": np.nanpercentile(particles, p, axis=0)
        for p in probs
    }


def _plot_observation_nowcast(y_obs, start_idx, obs_quant, save_path):
    x_hist = np.arange(len(y_obs))
    horizon = len(obs_quant["q50"])
    x_fut = np.arange(start_idx + 1, start_idx + 1 + horizon)
    colors = sns.color_palette("Paired")

    plt.figure(figsize=(12, 6))
    plt.plot(
        x_hist,
        y_obs,
        color=colors[3],
        marker="o",
        linewidth=0,
        markersize=3,
        alpha=0.6,
        label="Observed data",
    )
    plt.axvline(
        start_idx,
        color="tab:gray",
        linestyle="--",
        linewidth=1.5,
        label="Forecast start",
    )
    plt.fill_between(
        x_fut,
        obs_quant["q2_5"],
        obs_quant["q97_5"],
        color=colors[0],
        alpha=0.3,
        label="Nowcast 95% CI",
    )
    plt.fill_between(
        x_fut,
        obs_quant["q25"],
        obs_quant["q75"],
        color=colors[0],
        alpha=1.0,
        label="Nowcast 50% CI",
    )
    plt.plot(
        x_fut,
        obs_quant["q50"],
        color=colors[1],
        linestyle=":",
        linewidth=2.0,
        zorder=5,
        label="Nowcast median",
    )
    plt.xlabel("Week index")
    plt.ylabel("Observed weekly cases")
    plt.title("Sequential one-step-ahead nowcast of observed infections")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_nowcast(training_weeks, chains_dir, out_dir):
    chain_file = os.path.join(chains_dir, f"pmmh_truncated_w{training_weeks}.npz")
    save_plot = os.path.join(out_dir, f"posterior_observed_nowcast_sequential_w{training_weeks}.png")
    if not os.path.exists(chain_file):
        raise FileNotFoundError(f"Chain file not found: {chain_file}")

    chain = _load_chain(chain_file)
    samples = chain["samples"]
    fit_y_obs = chain["Y_obs"].astype(float)
    t_obs = len(fit_y_obs)
    if t_obs != training_weeks:
        raise ValueError(f"'{chain_file}' has {t_obs} observations, expected {training_weeks}.")

    _, full_y_obs = _load_weekly_dates(COVID_DATA_FILE, WEEKLY_GROUP_FREQ)
    if not np.array_equal(full_y_obs[:t_obs], fit_y_obs):
        raise ValueError("Chain observations do not match the prefix of the data file.")

    start_idx = training_weeks - 1
    out = _sequential_beta_nowcast(
        samples=samples,
        states_paths=chain["states_paths"],
        y_obs=full_y_obs,
        start_idx=start_idx,
        horizon_weeks=HORIZON_WEEKS,
        burn_in=int(samples.shape[0] * BURN_IN_FRACTION),
        num_particles=NUM_PARTICLES,
        seed=RANDOM_SEED,
        underreporting_factor=_load_underreporting_factor(chain_file),
        resample_ess_ratio=RESAMPLE_ESS_RATIO,
    )
    obs_quant = _quantile_summary(out["observed_pred_particles"])
    _plot_observation_nowcast(full_y_obs, start_idx, obs_quant, save_plot)
    print(f"Saved {save_plot}")


def main():
    parser = argparse.ArgumentParser(description="Sequential nowcasts from truncated PMMH chains.")
    parser.add_argument("--chains-dir", default="./chains")
    parser.add_argument("--out-dir", default="./figures")
    parser.add_argument("--training-weeks", type=int, nargs="+", default=TRAINING_WEEKS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for w in args.training_weeks:
        run_nowcast(w, args.chains_dir, args.out_dir)


if __name__ == "__main__":
    main()
