"""
PMMH (Particle Marginal Metropolis-Hastings) for SEIR Model

Implements MCMC for global parameters using SMC for marginal likelihood estimation.
Based on the seir_smc_final.py and Revised Functions notebook.

Key Features:
- Samples from joint posterior of parameters AND latent paths (β(t) and infections)
- Adaptive Metropolis with empirical covariance learning
- Automatic checkpoint saving every 100 iterations
- Returns full chain: parameters, log-likelihoods, beta paths, infection paths

PMMH Algorithm:
1. Propose new parameters (kappa, gamma, sigma, nu)
2. Run SMC to get marginal likelihood AND sample a latent path
3. Accept/reject via Metropolis-Hastings
4. Store accepted parameters and their corresponding latent path

This gives draws from the joint posterior p(theta, X | Y) where:
- theta = (kappa, gamma, sigma, nu)
- X = latent trajectories (beta(t), observed_infections(t), states(t))
- Y = observed weekly cases
"""

# Import all necesary libraries
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.special import logsumexp
import time
import datetime
import os
import json
import platform
import sys
import argparse


DEFAULT_PRIOR_SPECS = {
    "kappa": {
        "dist": "beta",
        "alpha": 9.0,
        "beta": 3.0,
    },
    "gamma": {
        "dist": "beta",
        "alpha": 10.0,
        "beta": 10.0,
    },
    "sigma": {
        "dist": "gamma",
        "shape": 1.0,
        "scale": 10.0,
    },
    "nu": {
        "dist": "gamma",
        "mean": 2.0,
        "interval_low": 1.5,
        "interval_high": 2.5,
        "mass_in_interval": 0.999999,
    },
}


def _gamma_prior_shape_for_mean_and_interval_mass(
    lower, upper, mean, mass
):
    def objective(shape):
        scale = mean / shape
        cdf_upper = stats.gamma(a=shape, scale=scale).cdf(upper)
        cdf_lower = stats.gamma(a=shape, scale=scale).cdf(lower)
        return (cdf_upper - cdf_lower) - mass

    return optimize.brentq(objective, 0.1, 1e5)


def _calibrate_gamma_prior_spec(name, prior_spec):
    if "shape" in prior_spec and ("scale" in prior_spec or "rate" in prior_spec):
        shape = float(prior_spec["shape"])
        if "scale" in prior_spec:
            scale = float(prior_spec["scale"])
            rate = 1.0 / scale
        else:
            rate = float(prior_spec["rate"])
            scale = 1.0 / rate

        return {
            "dist": "gamma",
            "mean": shape * scale,
            "shape": shape,
            "scale": scale,
            "rate": rate,
            "interval_low": prior_spec.get("interval_low"),
            "interval_high": prior_spec.get("interval_high"),
            "mass_in_interval": prior_spec.get("mass_in_interval"),
        }

    shape = _gamma_prior_shape_for_mean_and_interval_mass(
        prior_spec["interval_low"],
        prior_spec["interval_high"],
        prior_spec["mean"],
        prior_spec["mass_in_interval"],
    )
    scale = prior_spec["mean"] / shape
    rate = 1.0 / scale

    return {
        "dist": "gamma",
        "mean": prior_spec["mean"],
        "interval_low": prior_spec["interval_low"],
        "interval_high": prior_spec["interval_high"],
        "mass_in_interval": prior_spec["mass_in_interval"],
        "shape": shape,
        "scale": scale,
        "rate": rate,
    }


def _normalize_prior_spec(name, prior_spec):
    dist = prior_spec.get("dist", "gamma").lower()

    if dist == "beta":
        alpha = float(prior_spec["alpha"])
        beta = float(prior_spec["beta"])
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Beta prior hyperparameters must be positive for {name}.")
        return {
            "dist": "beta",
            "alpha": alpha,
            "beta": beta,
            "mean": alpha / (alpha + beta),
            "support": [0.0, 1.0],
        }

    if dist == "gamma":
        return _calibrate_gamma_prior_spec(name, prior_spec)

    raise ValueError(f"Unsupported prior distribution '{dist}' for parameter '{name}'.")


CALIBRATED_PRIOR_SPECS = {
    name: _normalize_prior_spec(name, spec)
    for name, spec in DEFAULT_PRIOR_SPECS.items()
}

# ============================================================================
# SEIR MODEL (same as before)
# ============================================================================


def BM_SEIR(Z_in, params, num_particles, N):
    """
    Weekly SEIR model with Brownian motion on log_beta
    """
    kappa, gamma, sigma = params[:3]

    S_t = Z_in[0, :]
    E_t = Z_in[1, :]
    I_t = Z_in[2, :]
    R_t = Z_in[3, :]
    log_beta_t = Z_in[4, :]

    # SEIR dynamics
    new_infected = np.exp(log_beta_t) * S_t * I_t / N
    observed_infections = kappa * E_t
    recovered = gamma * I_t

    # Update states (may go negative - will be caught by validity check)
    S_new = S_t - new_infected
    E_new = E_t + new_infected - observed_infections
    I_new = I_t + observed_infections - recovered
    R_new = R_t + recovered

    # Brownian motion update (h=1, so sqrt(h)=1)
    log_beta_new = log_beta_t + sigma * np.random.randn(num_particles)

    # Stack results
    Z_out = np.array([S_new, E_new, I_new, R_new, log_beta_new])

    return Z_out, observed_infections


# ============================================================================
# SEIR-SPECIFIC SMC CLASS (same as before)
# ============================================================================


class SEIRSMC:
    """SMC implementation for SEIR model"""

    def __init__(
        self,
        N,
        nu,
        model_params,
        Y_obs_initial,
        underreporting_factor,
        target_ess_ratio=0.9,
        enable_mcmc=True,
    ):
        self.N = N
        self.nu = nu
        self.model_params = model_params
        self.Y_obs_initial = Y_obs_initial
        self.target_ess_ratio = target_ess_ratio
        self.enable_mcmc = enable_mcmc
        self.underreporting_factor = underreporting_factor

    def initial_particles(self, num_particles):
        """Generate initial SEIR particles"""
        I0 = np.ones(num_particles) * self.Y_obs_initial
        E0 = np.random.uniform(0, 0.0001 * self.N, num_particles)
        R0 = np.zeros(num_particles)
        S0 = self.N - E0 - I0 - R0
        log_beta0 = np.log(np.random.uniform(0.01, 2, num_particles))
        return np.array([S0, E0, I0, R0, log_beta0])

    def propagate_and_likelihood(self, particles, observation):
        """Propagate particles and compute likelihoods"""
        num_particles = particles.shape[1]

        particles_next, weekly_infections = BM_SEIR(
            particles, self.model_params, num_particles, self.N
        )

        log_likelihoods = np.full(num_particles, -np.inf)
        valid_mask = (weekly_infections > 0) & np.isfinite(weekly_infections)

        if np.any(valid_mask):
            p = self.nu / (
                self.nu + weekly_infections[valid_mask] / self.underreporting_factor
            )
            p = np.clip(p, 1e-10, 1 - 1e-10)
            log_likelihoods[valid_mask] = stats.nbinom(
                n=self.nu, p=p
            ).logpmf(observation)

        return particles_next, weekly_infections, log_likelihoods

    def is_valid(self, particles, weekly_infections):
        """Check particle validity"""
        populations_valid = np.all(particles[:4, :] >= 0, axis=0)
        infections_valid = (weekly_infections >= 0) & np.isfinite(weekly_infections)
        return populations_valid & infections_valid

    def mcmc_step(
        self,
        particles_t,
        log_likelihoods,
        observation,
        alpha,
        step_size=0.1,
        return_accept_rate=False,
    ):
        """Vectorized MCMC jittering step for log_beta"""
        if not self.enable_mcmc:
            if return_accept_rate:
                return particles_t, log_likelihoods, 0.0
            else:
                return particles_t, log_likelihoods

        num_particles = particles_t.shape[1]

        proposed_log_beta = particles_t[-1, :] + np.random.normal(
            0, step_size, num_particles
        )
        # proposed_log_beta = np.clip(proposed_log_beta, -2, 2)

        proposed_particles = particles_t.copy()
        proposed_particles[-1, :] = proposed_log_beta

        proposed_next, proposed_infections = BM_SEIR(
            proposed_particles, self.model_params, num_particles, self.N
        )

        proposed_ll = np.full(num_particles, -np.inf)
        valid_mask = (
            (proposed_infections > 0)
            & np.isfinite(proposed_infections)
            & np.all(proposed_next[:4, :] >= 0, axis=0)
        )
        if np.any(valid_mask):
            p = self.nu / (
                self.nu + proposed_infections[valid_mask] / self.underreporting_factor
            )
            p = np.clip(p, 1e-10, 1 - 1e-10)
            proposed_ll[valid_mask] = stats.nbinom(
                n=self.nu, p=p
            ).logpmf(observation)

        # Compute log acceptance probability, handling -inf cases
        log_accept = np.full(num_particles, -np.inf)
        finite_mask = np.isfinite(proposed_ll) & np.isfinite(log_likelihoods)
        if np.any(finite_mask):
            log_accept[finite_mask] = alpha * (
                proposed_ll[finite_mask] - log_likelihoods[finite_mask]
            )
        # If proposed is finite but current is -inf, always accept
        accept_always = np.isfinite(proposed_ll) & ~np.isfinite(log_likelihoods)
        log_accept[accept_always] = np.inf

        accept = np.log(np.random.rand(num_particles)) < log_accept

        particles_t[-1, accept] = proposed_log_beta[accept]
        log_likelihoods[accept] = proposed_ll[accept]

        accept_rate = np.sum(accept) / num_particles
        if return_accept_rate:
            return particles_t, log_likelihoods, accept_rate
        else:
            return particles_t, log_likelihoods

    def find_next_alpha(self, alpha_prev, log_liks, tolerance=0.01, verbose=False):
        """Find next annealing parameter using bisection"""

        def compute_cess(alpha):
            if alpha <= alpha_prev:
                return len(log_liks)

            delta = alpha - alpha_prev
            a = delta * log_liks

            finite_mask = np.isfinite(a)
            if not np.any(finite_mask):
                return 0.0

            finite_liks = log_liks[finite_mask]
            if len(finite_liks) > 1 and np.var(finite_liks) < 1e-10:
                return len(log_liks)

            max_a = np.max(a[finite_mask])
            a_shifted = a[finite_mask] - max_a

            exp_a = np.exp(a_shifted)
            exp_2a = np.exp(2 * a_shifted)

            sum_exp_a = np.sum(exp_a)
            sum_exp_2a = np.sum(exp_2a)

            if sum_exp_a <= 0 or sum_exp_2a <= 0:
                return 0.0

            ess = (sum_exp_a**2) / sum_exp_2a
            return ess

        target_ess = self.target_ess_ratio * len(log_liks)
        current_ess = compute_cess(alpha_prev)
        ess_at_one = compute_cess(1.0)

        if ess_at_one > target_ess:
            alpha_target = min(alpha_prev + 0.1, 1.0)
            alpha_final = alpha_target
            converged = False
        else:
            alpha_low = alpha_prev
            alpha_high = 1.0
            converged = False

            for iteration in range(50):
                alpha_mid = (alpha_low + alpha_high) / 2
                ess_mid = compute_cess(alpha_mid)

                ess_ratio = ess_mid / len(log_liks)
                target_ratio = self.target_ess_ratio
                if abs(ess_ratio - target_ratio) < tolerance:
                    alpha_final = alpha_mid
                    converged = True
                    break

                if ess_mid > target_ess:
                    alpha_low = alpha_mid
                else:
                    alpha_high = alpha_mid

                if abs(alpha_high - alpha_low) < 1e-6:
                    break

            alpha_final = (alpha_low + alpha_high) / 2

        alpha_final = max(alpha_final, alpha_prev + 1e-6)
        alpha_final = min(alpha_final, 1.0)

        return alpha_final

    def systematic_resample(self, weights):
        """Systematic resampling"""
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform()) / n
        return np.searchsorted(np.cumsum(weights), positions)

    def annealing_step(
        self,
        particles_t,
        particles_tplus1,
        weekly_infections,
        log_likelihoods,
        observation,
        time_step=None,
    ):
        """
        Full annealing sequence from alpha=0 to alpha=1

        Returns the final resampling indices for ancestry tracking
        """
        alpha = 0.0
        log_Z = 0.0
        max_steps = 100

        cumulative_log_weights = np.zeros(len(log_likelihoods))
        cumulative_resample_indices = np.arange(len(log_likelihoods))

        valid_mask = self.is_valid(particles_tplus1, weekly_infections)
        if not np.any(valid_mask):
            return (
                particles_t,
                particles_tplus1,
                weekly_infections,
                log_likelihoods,
                -np.inf,
                None,
            )

        log_likelihoods[~valid_mask] = -np.inf

        for step in range(max_steps):
            if 1.0 - alpha < 1e-4:
                break

            alpha_new = self.find_next_alpha(alpha, log_likelihoods, verbose=False)
            delta = alpha_new - alpha

            incremental_log_weights = delta * log_likelihoods

            if not np.any(np.isfinite(incremental_log_weights)):
                break

            log_Z += logsumexp(incremental_log_weights) - np.log(
                len(incremental_log_weights)
            )

            cumulative_log_weights = alpha_new * log_likelihoods
            weights = np.exp(cumulative_log_weights - logsumexp(cumulative_log_weights))
            ess = 1.0 / np.sum(weights**2)

            if ess < len(weights) * 0.5:
                indices = self.systematic_resample(weights)
                particles_t = particles_t[:, indices]
                particles_tplus1 = particles_tplus1[:, indices]
                weekly_infections = weekly_infections[indices]
                log_likelihoods = log_likelihoods[indices]
                cumulative_log_weights = np.zeros(len(log_likelihoods))
                # Track cumulative resampling for ancestry
                cumulative_resample_indices = cumulative_resample_indices[indices]

            particles_t, log_likelihoods, accept_rate = self.mcmc_step(
                particles_t,
                log_likelihoods,
                observation,
                alpha_new,
                return_accept_rate=True,
            )

            alpha = alpha_new

        # Force final resampling to eliminate particles with low likelihood
        cumulative_log_weights = alpha * log_likelihoods
        if np.any(np.isfinite(cumulative_log_weights)):
            weights = np.exp(cumulative_log_weights - logsumexp(cumulative_log_weights))
            indices = self.systematic_resample(weights)
            particles_t = particles_t[:, indices]
            log_likelihoods = log_likelihoods[indices]
            # Update cumulative indices
            cumulative_resample_indices = cumulative_resample_indices[indices]

        particles_tplus1, weekly_infections, log_likelihoods = (
            self.propagate_and_likelihood(particles_t, observation)
        )

        return (
            particles_t,
            particles_tplus1,
            weekly_infections,
            log_likelihoods,
            log_Z,
            cumulative_resample_indices,
        )

    def run(self, Y_obs, num_particles, return_path=False):
        """
        Run SEIR SMC for full time series

        Args:
            Y_obs: Observed data
            num_particles: Number of particles
            return_path: If True, return a sampled path from the posterior

        Returns:
            log_marginal_likelihood: Log marginal likelihood
            If return_path=True, also returns:
                beta_path: Sampled beta trajectory (T,)
                infection_path: Sampled infection trajectory (T,)
        """
        T = len(Y_obs)
        particles = self.initial_particles(num_particles)
        total_log_marginal_likelihood = 0.0

        # Store particles and ancestry for path sampling
        if return_path:
            particles_history = []
            infections_history = []
            ancestry_matrix = np.zeros((num_particles, T), dtype=int)

            particles_history.append(particles.copy())
            infections_history.append(np.ones(num_particles) * self.Y_obs_initial)
            ancestry_matrix[:, 0] = np.arange(num_particles)  # Initial ancestry

        for t in range(1, T):
            particles_next, weekly_infections, log_likelihoods = (
                self.propagate_and_likelihood(particles, Y_obs[t])
            )

            (
                particles,
                particles_next,
                weekly_infections,
                log_likelihoods,
                log_Z,
                resample_indices,
            ) = self.annealing_step(
                particles,
                particles_next,
                weekly_infections,
                log_likelihoods,
                Y_obs[t],
                time_step=t,
            )

            particles = particles_next
            total_log_marginal_likelihood += log_Z

            if return_path:
                particles_history.append(particles.copy())
                infections_history.append(weekly_infections.copy())

                # Update ancestry matrix using resampling indices
                if resample_indices is not None:
                    # Reshuffle ancestry: ancestry_matrix[:,1:t-1] = ancestry_matrix[resample_indices,1:t-1]
                    ancestry_matrix[:, :t] = ancestry_matrix[resample_indices, :t]
                    ancestry_matrix[:, t] = resample_indices
                else:
                    ancestry_matrix[:, t] = np.arange(num_particles)

        if return_path:
            # Sample a coherent trajectory using ancestry matrix
            beta_path, infection_path, states_path = self.sample_path_with_ancestry(
                particles_history, infections_history, ancestry_matrix
            )

            return total_log_marginal_likelihood, beta_path, infection_path, states_path
        else:
            return total_log_marginal_likelihood

    def sample_path_with_ancestry(
        self, particles_history, infections_history, ancestry_matrix
    ):
        """
        Sample a single coherent trajectory using the ancestry matrix

        Algorithm:
        1. Pick a final particle uniformly
        2. Trace its ancestry backwards through the matrix
        3. Extract the corresponding values at each time step
        """
        T = len(particles_history)
        num_particles = particles_history[0].shape[1]

        # Pick final particle uniformly
        final_idx = np.random.randint(0, num_particles)

        beta_path = np.zeros(T)
        infection_path = np.zeros(T)
        states_path = np.zeros((5, T))

        # Trace lineage backwards through ancestry matrix
        current_idx = final_idx
        for t in range(T - 1, -1, -1):
            # Extract values for this particle at time t
            beta_path[t] = np.exp(particles_history[t][4, current_idx])
            infection_path[t] = infections_history[t][current_idx]
            states_path[:, t] = particles_history[t][:, current_idx]

            # Move to ancestor at previous time (if not at t=0)
            if t > 0:
                # Find which particle at time t-1 led to current_idx at time t
                # This is stored in ancestry_matrix[current_idx, t-1]
                current_idx = ancestry_matrix[current_idx, t - 1]

        return beta_path, infection_path, states_path


# ============================================================================
# PMMH ALGORITHM
# ============================================================================


class PMMH:
    """
    Particle Marginal Metropolis-Hastings for SEIR model

    Estimates posterior distribution of global parameters (kappa, gamma, sigma, nu)
    using SMC to compute marginal likelihood.
    """

    def __init__(
        self,
        Y_obs,
        N,
        underreporting_factor,
        num_particles=1000,
        target_ess_ratio=0.9,
        prior_specs=None,
    ):
        """
        Args:
            Y_obs: Observed weekly case counts
            N: Total population size
            underreporting_factor: Divisor applied to modeled infections to get the
                mean of the observation model (accounts for case underreporting)
            num_particles: Number of SMC particles
            target_ess_ratio: Target ESS ratio for SMC annealing
            prior_specs: Optional dict of Gamma prior specs keyed by parameter name
        """
        self.Y_obs = Y_obs
        self.N = N
        self.num_particles = num_particles
        self.target_ess_ratio = target_ess_ratio
        self.underreporting_factor = underreporting_factor
        if prior_specs is None:
            self.prior_specs = {
                name: dict(spec) for name, spec in CALIBRATED_PRIOR_SPECS.items()
            }
        else:
            self.prior_specs = {
                name: dict(spec) for name, spec in CALIBRATED_PRIOR_SPECS.items()
            }
            for name, spec in prior_specs.items():
                if name not in self.prior_specs:
                    raise KeyError(
                        f"Unknown prior parameter '{name}'. Expected one of {list(self.prior_specs.keys())}."
                    )
                self.prior_specs[name] = _normalize_prior_spec(name, spec)

    def prior_log_pdf(self, params):
        """
        Log prior density for global parameters
        params = [kappa, gamma, sigma, nu]

        """
        param_order = ["kappa", "gamma", "sigma", "nu"]

        if np.any(np.asarray(params) <= 0):
            return -np.inf

        log_prior = 0.0
        for value, name in zip(params, param_order):
            prior_spec = self.prior_specs[name]
            if prior_spec["dist"] == "beta":
                log_prior += stats.beta(
                    a=prior_spec["alpha"], b=prior_spec["beta"]
                ).logpdf(value)
            elif prior_spec["dist"] == "gamma":
                log_prior += stats.gamma(
                    a=prior_spec["shape"], scale=prior_spec["scale"]
                ).logpdf(value)
            else:
                return -np.inf

        return log_prior

    def get_prior_spec(self):
        """Return a serializable specification of the priors used in prior_log_pdf."""
        return {
            "version": "2.0",
            "param_order": ["kappa", "gamma", "sigma", "nu"],
            **{name: dict(spec) for name, spec in self.prior_specs.items()},
        }

    def build_run_metadata(
        self,
        num_iterations,
        initial_params,
        step_sizes,
        adapt_proposal,
        adapt_every,
        adapt_start,
        adapt_stop,
        adapt_gamma_power,
        adapt_shrinkage,
        adapt_ridge,
        min_step_sizes,
        save_every,
        save_file,
    ):
        """Build a serializable metadata dict for reproducibility."""
        return {
            "created_at": datetime.datetime.now().isoformat(),
            "script": os.path.basename(__file__) if "__file__" in globals() else "interactive",
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
            "model_specification": {
                "model": "SEIR with Brownian motion on log_beta",
                "state_vector": ["S", "E", "I", "R", "log_beta"],
                "time_step": "weekly",
                "transitions": {
                    "new_infected": "exp(log_beta_t) * S_t * I_t / N",
                    "observed_infections": "kappa * E_t",
                    "recovered": "gamma * I_t",
                },
                "observation_model": {
                    "distribution": "Negative Binomial",
                    "mean": "weekly_infections",
                    "overdispersion_param": "nu",
                    "underreporting_factor": float(self.underreporting_factor),
                },
                "fixed_population_N": float(self.N),
                "num_observations": int(len(self.Y_obs)),
                "initial_observation": float(self.Y_obs[0]),
            },
            "prior_specification": self.get_prior_spec(),
            "particle_filter_specification": {
                "algorithm": "annealed SMC",
                "num_particles": int(self.num_particles),
                "target_ess_ratio": float(self.target_ess_ratio),
                "resampling": "systematic",
                "resample_threshold_ess_ratio": 0.5,
                "max_annealing_steps": 100,
                "particle_mcmc_jitter_enabled": True,
            },
            "mcmc_specification": {
                "algorithm": "Particle Marginal Metropolis-Hastings",
                "num_iterations": int(num_iterations),
                "initial_params": np.asarray(initial_params, dtype=float).tolist(),
                "proposal": "multivariate Gaussian random walk",
                "initial_step_sizes": np.asarray(step_sizes, dtype=float).tolist(),
                "adapt_proposal": bool(adapt_proposal),
                "adapt_every": int(adapt_every),
                "adapt_start": int(adapt_start),
                "adapt_stop": int(adapt_stop),
                "adapt_gamma_power": float(adapt_gamma_power),
                "adapt_shrinkage": float(adapt_shrinkage),
                "adapt_ridge": float(adapt_ridge),
                "min_step_sizes": np.asarray(min_step_sizes, dtype=float).tolist(),
            },
            "outputs": {
                "chain_file": save_file,
            },
        }

    def proposal(self, current_params, proposal_cov):
        """
        Multivariate random walk proposal for parameters

        Args:
            current_params: Current parameter values [kappa, gamma, sigma, nu]
            proposal_cov: Proposal covariance matrix
        """
        proposed = np.random.multivariate_normal(current_params, proposal_cov)

        # Clip to valid ranges (0, 1)
        # proposed[0] = np.clip(proposed[0], 0.0001, 0.9999)  # kappa
        # proposed[1] = np.clip(proposed[1], 0.0001, 0.9999)  # gamma
        # proposed[2] = np.clip(proposed[2], 0.0001, 0.9999)  # sigma
        # proposed[3] = np.clip(proposed[3], 0.0001, 0.9999)  # nu

        return proposed

    def run_smc(self, params, return_path=False):
        """
        Run SMC with given parameters

        Args:
            params: [kappa, gamma, sigma, nu]

        Returns:
            log_marginal_likelihood: Estimate from SMC
            If return_path=True, also returns beta_path and infection_path
        """
        kappa, gamma, sigma, nu = params
        model_params = np.array([kappa, gamma, sigma])

        smc = SEIRSMC(
            N=self.N,
            nu=nu,
            model_params=model_params,
            Y_obs_initial=self.Y_obs[0],
            target_ess_ratio=self.target_ess_ratio,
            enable_mcmc=True,
            underreporting_factor=self.underreporting_factor,
        )

        if return_path:
            log_marginal_likelihood, beta_path, infection_path, states_path = smc.run(
                self.Y_obs, self.num_particles, return_path=True
            )
            return log_marginal_likelihood, beta_path, infection_path, states_path
        else:
            log_marginal_likelihood = smc.run(
                self.Y_obs, self.num_particles, return_path=False
            )
            return log_marginal_likelihood

    def run(
        self,
        num_iterations,
        initial_params=None,
        step_sizes=None,
        adapt_proposal=False,
        adapt_every=1000,
        adapt_start=2000,
        adapt_stop=None,
        adapt_gamma_power=0.6,
        adapt_shrinkage=0.05,
        adapt_ridge=1e-5,
        min_step_sizes=None,
        verbose=True,
        save_every=100,
        save_file="pmmh_chain.npz",
        metadata_file=None,
    ):
        """
        Run PMMH algorithm with random walk proposals

        Uses random walk Metropolis-Hastings proposals with optional adaptive
        covariance updates from empirical chain history.

        PMMH also samples latent paths: each accepted iteration includes a draw
        from the posterior distribution over beta(t) and infection(t) trajectories.

        Args:
            num_iterations: Number of MCMC iterations
            initial_params: Starting values [kappa, gamma, sigma, nu]
            step_sizes: Initial marginal proposal std devs for [kappa, gamma, sigma, nu]
            adapt_proposal: If True, update the full proposal covariance from chain history
            adapt_every: Adapt proposal every this many iterations
            adapt_start: First iteration at which adaptation is allowed
            adapt_stop: Last iteration at which adaptation is allowed (None = all)
            adapt_gamma_power: Exponent for diminishing adaptation rate k^{-power}
            adapt_shrinkage: Shrinkage weight toward initial covariance in each adaptation
            adapt_ridge: Ridge regularization added to diagonal covariance entries
            min_step_sizes: Lower bounds for adaptive marginal proposal std devs
            verbose: Print progress
            save_every: Save chain to file every N iterations (default: 100)
            save_file: Filename for saving chain (default: 'pmmh_chain.npz')

        Returns:
            samples: Array of parameter samples (num_iterations x 4)
            log_likelihoods: Log marginal likelihoods for each sample
            beta_paths: Array of sampled beta trajectories (num_iterations x T)
            infection_paths: Array of sampled infection trajectories (num_iterations x T)
            acceptance_rate: Overall acceptance rate
        """
        # Initialize
        if save_file is None or os.path.exists(save_file):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"pmmh_chain_{timestamp}.npz"
        if metadata_file is None:
            save_root, _ = os.path.splitext(save_file)
            metadata_file = f"{save_root}_metadata.json"

        if initial_params is None:
            initial_params = np.array([0.5, 0.5, 0.5, 1.5])

        if step_sizes is None:
            # Baseline marginal std devs for the initial diagonal proposal covariance
            step_sizes = np.array(
                [0.05, 0.1, 0.05, 0.05]
            )  # [kappa, gamma, sigma, nu]

        step_sizes = np.asarray(step_sizes, dtype=float)
        if min_step_sizes is None:
            min_step_sizes = np.array([0.005, 0.005, 0.01, 0.01], dtype=float)
        else:
            min_step_sizes = np.asarray(min_step_sizes, dtype=float)

        if adapt_stop is None:
            adapt_stop = num_iterations
        adapt_start = int(max(2, adapt_start))
        adapt_stop = int(max(adapt_start, adapt_stop))
        adapt_every = int(max(1, adapt_every))
        adapt_gamma_power = float(max(0.5, adapt_gamma_power))
        adapt_shrinkage = float(np.clip(adapt_shrinkage, 0.0, 1.0))
        adapt_ridge = float(max(0.0, adapt_ridge))

        initial_proposal_cov = np.diag(step_sizes**2)
        min_proposal_cov = np.diag(min_step_sizes**2)
        current_proposal_cov = initial_proposal_cov.copy()
        proposal_scale = (2.38**2) / len(step_sizes)

        def regularize_covariance(cov_matrix):
            cov_matrix = np.asarray(cov_matrix, dtype=float)
            cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * adapt_ridge
            return cov_matrix

        prior_spec_json = json.dumps(self.get_prior_spec())
        run_metadata = self.build_run_metadata(
            num_iterations=num_iterations,
            initial_params=initial_params,
            step_sizes=step_sizes,
            adapt_proposal=adapt_proposal,
            adapt_every=adapt_every,
            adapt_start=adapt_start,
            adapt_stop=adapt_stop,
            adapt_gamma_power=adapt_gamma_power,
            adapt_shrinkage=adapt_shrinkage,
            adapt_ridge=adapt_ridge,
            min_step_sizes=min_step_sizes,
            save_every=save_every,
            save_file=save_file,
        )

        def write_metadata(iteration, accepted, current_log_lik, current_params, proposal_cov):
            elapsed_seconds = time.time() - start_time
            metadata = dict(run_metadata)
            metadata["outputs"] = {
                **run_metadata["outputs"],
                "metadata_file": metadata_file,
                "diagnostics_files": ["pmmh_diagnostics.png", "latent_paths.png"],
            }
            metadata["runtime_status"] = {
                "iteration": int(iteration),
                "accepted": int(accepted),
                "acceptance_rate": float(accepted / max(1, iteration)),
                "elapsed_seconds": float(elapsed_seconds),
                "current_log_likelihood": float(current_log_lik),
                "current_parameters": np.asarray(current_params, dtype=float).tolist(),
                "current_proposal_covariance": np.asarray(proposal_cov, dtype=float).tolist(),
            }
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

        T = len(self.Y_obs)
        samples = np.zeros((num_iterations, 4))
        log_likelihoods = np.zeros(num_iterations)
        beta_paths = np.zeros((num_iterations, T))
        infection_paths = np.zeros((num_iterations, T))
        states_paths = np.zeros((num_iterations, 5, T))
        accepted = 0

        # Initialize chain
        current_params = initial_params.copy()
        current_log_prior = self.prior_log_pdf(current_params)

        if verbose:
            print("Running initial SMC to get starting log-likelihood and path...")
        (
            current_log_lik,
            current_beta_path,
            current_infection_path,
            current_states_path,
        ) = self.run_smc(current_params, return_path=True)
        current_log_posterior = current_log_prior + current_log_lik

        if verbose:
            print(f"Initial log-likelihood: {current_log_lik:.4f}")
            print(f"Initial log-prior: {current_log_prior:.4f}")
            print(
                f"Initial parameters: κ={current_params[0]:.3f}, γ={current_params[1]:.3f}, σ={current_params[2]:.3f}, ν={current_params[3]:.3f}"
            )
            print(f"Initial proposal std devs: {step_sizes}")
            print(f"\nStarting PMMH with {num_iterations} iterations...")
            if adapt_proposal:
                print(
                    f"Using adaptive multivariate random walk proposals (covariance updated every {adapt_every} iterations)"
                )
                print(
                    f"Adaptation window: iterations {adapt_start} to {adapt_stop} | gamma power: {adapt_gamma_power:.2f} | shrinkage: {adapt_shrinkage:.2f}"
                )
            else:
                print("Using random walk proposals (no adaptation)")
            print(f"Saving chain every {save_every} iterations to '{save_file}'")
            print("=" * 70)

        start_time = time.time()

        for iteration in range(num_iterations):
            # Simple random walk proposal
            proposed_params = self.proposal(current_params, current_proposal_cov)
            proposed_log_prior = self.prior_log_pdf(proposed_params)

            if proposed_log_prior == -np.inf:
                # Reject immediately if outside prior support
                samples[iteration] = current_params
                log_likelihoods[iteration] = current_log_lik
                beta_paths[iteration] = current_beta_path
                infection_paths[iteration] = current_infection_path
                states_paths[iteration] = current_states_path
            else:
                # Run SMC with proposed parameters - always sample a path
                (
                    proposed_log_lik,
                    proposed_beta_path,
                    proposed_infection_path,
                    proposed_states_path,
                ) = self.run_smc(proposed_params, return_path=True)
                proposed_log_posterior = proposed_log_prior + proposed_log_lik

                # Metropolis-Hastings acceptance
                log_accept_ratio = proposed_log_posterior - current_log_posterior

                if np.log(np.random.rand()) < log_accept_ratio:
                    # Accept: update parameters AND paths
                    current_params = proposed_params
                    current_log_prior = proposed_log_prior
                    current_log_lik = proposed_log_lik
                    current_log_posterior = proposed_log_posterior
                    current_beta_path = proposed_beta_path
                    current_infection_path = proposed_infection_path
                    current_states_path = proposed_states_path
                    accepted += 1

                samples[iteration] = current_params
                log_likelihoods[iteration] = current_log_lik
                beta_paths[iteration] = current_beta_path
                infection_paths[iteration] = current_infection_path
                states_paths[iteration] = current_states_path

            # Adapt proposal scales from empirical parameter history.
            if (
                adapt_proposal
                and (iteration + 1) % adapt_every == 0
                and adapt_start <= (iteration + 1) <= adapt_stop
            ):
                history = samples[: iteration + 1]
                empirical_cov = np.cov(history, rowvar=False, ddof=1)
                empirical_cov = np.atleast_2d(empirical_cov)
                base_cov = proposal_scale * empirical_cov + min_proposal_cov
                target_cov = (1.0 - adapt_shrinkage) * base_cov + adapt_shrinkage * initial_proposal_cov
                target_cov = regularize_covariance(target_cov)

                adapt_count = ((iteration + 1) - adapt_start) // adapt_every + 1
                gamma_n = adapt_count ** (-adapt_gamma_power)
                blended_cov = (1.0 - gamma_n) * current_proposal_cov + gamma_n * target_cov
                current_proposal_cov = regularize_covariance(blended_cov)

                if verbose:
                    print(
                        f"  [Adapt] Iter {iteration + 1}: gamma={gamma_n:.4f}, proposal covariance ->\n{np.array2string(current_proposal_cov, precision=5)}"
                    )

            # Save chain periodically
            if (iteration + 1) % save_every == 0:
                np.savez_compressed(
                    save_file,
                    samples=samples[: iteration + 1],
                    log_likelihoods=log_likelihoods[: iteration + 1],
                    beta_paths=beta_paths[: iteration + 1],
                    infection_paths=infection_paths[: iteration + 1],
                    states_paths=states_paths[: iteration + 1],
                    accepted=accepted,
                    iteration=iteration + 1,
                    Y_obs=self.Y_obs,
                    prior_spec_json=prior_spec_json,
                )
                write_metadata(
                    iteration=iteration + 1,
                    accepted=accepted,
                    current_log_lik=current_log_lik,
                    current_params=current_params,
                    proposal_cov=current_proposal_cov,
                )
                if verbose:
                    print(f"  [Save] Chain saved at iteration {iteration + 1}")

            # Print progress
            print_freq = 100
            if verbose and (iteration + 1) % print_freq == 0:
                elapsed = time.time() - start_time
                accept_rate = accepted / (iteration + 1)
                progress = (iteration + 1) / num_iterations * 100
                eta = elapsed / (iteration + 1) * (num_iterations - iteration - 1)
                print(
                    f"[{progress:5.1f}%] Iter {iteration + 1:5d}/{num_iterations} | Accept: {accept_rate:.3f} | "
                    f"Log-lik: {current_log_lik:7.2f} | "
                    f"κ={current_params[0]:.3f} γ={current_params[1]:.3f} σ={current_params[2]:.3f} ν={current_params[3]:.3f} | "
                    f"ETA: {eta:.0f}s"
                )

        acceptance_rate = accepted / num_iterations

        # Final save
        np.savez_compressed(
            save_file,
            samples=samples,
            log_likelihoods=log_likelihoods,
            beta_paths=beta_paths,
            infection_paths=infection_paths,
            states_paths=states_paths,
            accepted=accepted,
            iteration=num_iterations,
            Y_obs=self.Y_obs,
            prior_spec_json=prior_spec_json,
        )
        write_metadata(
            iteration=num_iterations,
            accepted=accepted,
            current_log_lik=current_log_lik,
            current_params=current_params,
            proposal_cov=current_proposal_cov,
        )

        if verbose:
            print("=" * 70)
            print(f"PMMH completed in {time.time() - start_time:.2f} seconds")
            print(f"Overall acceptance rate: {acceptance_rate:.3f}")
            print(f"Final chain saved to '{save_file}'")
            print(f"Run metadata saved to '{metadata_file}'")

        return (
            samples,
            log_likelihoods,
            beta_paths,
            infection_paths,
            states_paths,
            acceptance_rate,
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Run PMMH for the SEIR model.")
    p.add_argument("--eta", type=float, required=True,
                   help="Underreporting factor (divisor of modeled infections).")
    p.add_argument("--max-weeks", type=int, default=None,
                   help="Fit only the first N weeks (default: all weeks).")
    p.add_argument("--num-iterations", type=int, default=100_000)
    p.add_argument("--num-particles", type=int, default=5_000)
    p.add_argument("--save-file", required=True, help="Output .npz chain file.")
    return p.parse_args()


USER_CONFIG = {
    "data": {
        "covid_data_file": "./data/covid_df.feather",
        "population_file": "./data/INEGI_2020_State_Population.feather",
        "weekly_group_freq": "W-SUN",
    },
    "priors": {
        "kappa": {"dist": "beta", "alpha": 9.0, "beta": 3.0},
        "gamma": {"dist": "beta", "alpha": 10.0, "beta": 10.0},
        "sigma": {"dist": "gamma", "shape": 1.0, "scale": 10.0},
        "nu": {
            "dist": "gamma",
            "mean": 2.0,
            "interval_low": 1.5,
            "interval_high": 2.5,
            "mass_in_interval": 0.999999,
        },
    },
    "sampler": {"target_ess_ratio": 0.9},
    "mcmc": {
        "initial_params": [0.5, 0.5, 0.5, 1.5],
        "step_sizes": [0.05, 0.1, 0.05, 0.05],
        "adapt_proposal": False,
        "adapt_every": 1000,
        "adapt_start": 2000,
        "adapt_stop": None,
        "adapt_gamma_power": 0.6,
        "adapt_shrinkage": 0.05,
        "adapt_ridge": 1e-5,
        "min_step_sizes": [0.005, 0.005, 0.01, 0.01],
        "verbose": True,
        "save_every": 100,
        "metadata_file": None,
    },
}


if __name__ == "__main__":
    args = parse_args()
    cfg = USER_CONFIG

    data = pd.read_feather(cfg["data"]["covid_data_file"])
    weekly_covid_df = (
        data.groupby([pd.Grouper(key="date", freq=cfg["data"]["weekly_group_freq"])])
        .agg({"confirmed_cases": "sum"})
        .reset_index()
    )
    N = pd.read_feather(cfg["data"]["population_file"])["population"].sum()
    Y_obs = weekly_covid_df["confirmed_cases"].values[: args.max_weeks]

    print(f"Population: {N:,} | weeks: {len(Y_obs)} | eta: {args.eta}")

    pmmh = PMMH(
        Y_obs,
        N,
        num_particles=args.num_particles,
        target_ess_ratio=cfg["sampler"]["target_ess_ratio"],
        prior_specs=cfg["priors"],
        underreporting_factor=args.eta,
    )

    mcmc_cfg = dict(cfg["mcmc"])
    for key in ("initial_params", "step_sizes", "min_step_sizes"):
        mcmc_cfg[key] = np.asarray(mcmc_cfg[key], dtype=float)
    mcmc_cfg["num_iterations"] = args.num_iterations
    mcmc_cfg["save_file"] = args.save_file

    pmmh.run(**mcmc_cfg)
