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
1. Propose new parameters (kappa, gamma, sigma, overdispersion)
2. Run SMC to get marginal likelihood AND sample a latent path
3. Accept/reject via Metropolis-Hastings
4. Store accepted parameters and their corresponding latent path

This gives draws from the joint posterior p(theta, X | Y) where:
- theta = (kappa, gamma, sigma, overdispersion)
- X = latent trajectories (beta(t), infections(t))
- Y = observed weekly cases
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import logsumexp
import time


# ============================================================================
# SEIR MODEL (same as before)
# ============================================================================

def BM_SEIR(V_in, params, num_particles, N, m):
    """
    Optimized SEIR model with Brownian motion on log_beta
    
    Optimized for m=0 case (single time step, no sub-stepping)
    which is the most common usage in this code.
    """
    kappa, gamma, sigma = params[:3]
    
    # Fast path for m=0 (single time step, h=1)
    if m == 0:
        S_t = V_in[0, :]
        E_t = V_in[1, :]
        I_t = V_in[2, :]
        R_t = V_in[3, :]
        log_beta_t = V_in[4, :]
        
        # NO CLIPPING - use raw log_beta values, let MCMC handle invalid proposals
        log_beta_capped = log_beta_t
        
        # SEIR dynamics (h=1, so no multiplication by h)
        infections = np.exp(log_beta_capped) * S_t * I_t / N
        latent = kappa * E_t
        recovered = gamma * I_t
        
        # Update states (may go negative - will be caught by validity check)
        S_new = S_t - infections
        E_new = E_t + infections - latent
        I_new = I_t + latent - recovered
        R_new = R_t + recovered
        
        # Brownian motion update (h=1, so sqrt(h)=1)
        log_beta_new = log_beta_t + sigma * np.random.randn(num_particles)
        
        # Stack results
        V_out = np.array([S_new, E_new, I_new, R_new, log_beta_new])
        
        return V_out, infections
    
    # General case for m > 0 (sub-stepping with Euler-Maruyama)
    else:
        num_steps = m + 1  
        h = 1 / num_steps 

        V = np.zeros([V_in.shape[0], num_particles, num_steps + 1])    
        new_infected = np.zeros([num_particles])

        V[:,:,0] = V_in

        for t in range(1, num_steps + 1):
            # NO CLIPPING - use raw log_beta values, let MCMC handle invalid proposals
            S_t = V[0, :, t-1]
            I_t = V[2, :, t-1]
            
            log_beta_capped = V[4, :, t-1]
            
            infections = np.exp(log_beta_capped) * S_t * I_t / N
            latent = kappa * V[1, :, t-1]
            recovered = gamma * V[2, :, t-1]
            
            V[0, :, t] = V[0, :, t-1] - infections*h
            V[1, :, t] = V[1, :, t-1] + (infections  - latent)*h
            V[2, :, t] = V[2, :, t-1] + (latent - recovered)*h
            V[3, :, t] = V[3, :, t-1] + recovered*h
            
            # NO CLIPPING - compartments may go negative, caught by validity check
            
            dB = np.random.randn(num_particles)
            V[4,:,t] = V[4,:,t-1] + sigma * np.sqrt(h) * dB
            
            new_infected += infections*h
        
        return V[:,:,-1], new_infected


# ============================================================================
# SEIR-SPECIFIC SMC CLASS (same as before)
# ============================================================================

class SEIRSMC:
    """SMC implementation for SEIR model"""
    
    def __init__(self, N, overdispersion, model_params, Y_obs_initial, 
                 target_ess_ratio=0.9, enable_mcmc=True): 
        self.N = N
        self.overdispersion = overdispersion
        self.model_params = model_params
        self.Y_obs_initial = Y_obs_initial
        self.target_ess_ratio = target_ess_ratio
        self.enable_mcmc = enable_mcmc
    
    def initial_particles(self, num_particles):
        """Generate initial SEIR particles"""
        I0 = np.ones(num_particles) * self.Y_obs_initial
        E0 = np.random.uniform(0, 0.0001 * self.N, num_particles)
        R0 = np.zeros(num_particles)
        S0 = self.N - E0 - I0 - R0
        log_beta0 = np.log(np.random.uniform(0.01, 2, num_particles))
        return np.array([S0, E0, I0, R0, log_beta0])
    
    def propagate_and_likelihood(self, particles, observation, m=0):
        """Propagate particles and compute likelihoods"""
        num_particles = particles.shape[1]
        
        particles_next, weekly_infections = BM_SEIR(
            particles, self.model_params, num_particles, self.N, m
        )
        
        log_likelihoods = np.full(num_particles, -np.inf)
        valid_mask = (weekly_infections > 0) & np.isfinite(weekly_infections)
        
        if np.any(valid_mask):
            # Use correct negative binomial parameterization from notebook
            p = 1 / (1 + self.overdispersion * weekly_infections[valid_mask])
            p = np.clip(p, 1e-10, 1-1e-10)
            log_likelihoods[valid_mask] = stats.nbinom(n=1/self.overdispersion, p=p).logpmf(observation)
        
        return particles_next, weekly_infections, log_likelihoods
    
    def is_valid(self, particles, weekly_infections):
        """Check particle validity"""
        populations_valid = np.all(particles[:4, :] >= 0, axis=0)
        infections_valid = (weekly_infections >= 0) & np.isfinite(weekly_infections)
        return populations_valid & infections_valid
    
    def mcmc_step(self, particles_t, log_likelihoods, observation, alpha, step_size=0.1, return_accept_rate=False):
        """Vectorized MCMC jittering step for log_beta"""
        if not self.enable_mcmc:
            if return_accept_rate:
                return particles_t, log_likelihoods, 0.0
            else:
                return particles_t, log_likelihoods
        
        num_particles = particles_t.shape[1]
        
        proposed_log_beta = particles_t[-1, :] + np.random.normal(0, step_size, num_particles)
        # proposed_log_beta = np.clip(proposed_log_beta, -2, 2)
        
        proposed_particles = particles_t.copy()
        proposed_particles[-1, :] = proposed_log_beta
        
        proposed_next, proposed_infections = BM_SEIR(
            proposed_particles, self.model_params, num_particles, self.N, 0
        )
        
        valid_mask = (proposed_infections > 0) & np.isfinite(proposed_infections) & np.all(proposed_next[:4, :] >= 0, axis=0)
        
        proposed_ll = np.full(num_particles, -np.inf)
        if np.any(valid_mask):
            p = 1 / (1 + self.overdispersion * proposed_infections[valid_mask])
            p = np.clip(p, 1e-10, 1-1e-10)
            proposed_ll[valid_mask] = stats.nbinom(n=1/self.overdispersion, p=p).logpmf(observation)
        
        # Compute log acceptance probability, handling -inf cases
        log_accept = np.full(num_particles, -np.inf)
        finite_mask = np.isfinite(proposed_ll) & np.isfinite(log_likelihoods)
        if np.any(finite_mask):
            log_accept[finite_mask] = alpha * (proposed_ll[finite_mask] - log_likelihoods[finite_mask])
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
            
            ess = (sum_exp_a ** 2) / sum_exp_2a
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
    
    def annealing_step(self, particles_t, particles_tplus1, weekly_infections, 
                       log_likelihoods, observation, time_step=None):
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
            return particles_t, particles_tplus1, weekly_infections, log_likelihoods, -np.inf, None
        
        log_likelihoods[~valid_mask] = -np.inf
        
        for step in range(max_steps):
            if 1.0 - alpha < 1e-4:
                break
            
            alpha_new = self.find_next_alpha(alpha, log_likelihoods, verbose=False)
            delta = alpha_new - alpha
            
            incremental_log_weights = delta * log_likelihoods
            
            if not np.any(np.isfinite(incremental_log_weights)):
                break
                
            log_Z += logsumexp(incremental_log_weights) - np.log(len(incremental_log_weights))
            
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
                particles_t, log_likelihoods, observation, alpha_new, return_accept_rate=True
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
        
        particles_tplus1, weekly_infections, log_likelihoods = self.propagate_and_likelihood(
            particles_t, observation
        )
        
        return particles_t, particles_tplus1, weekly_infections, log_likelihoods, log_Z, cumulative_resample_indices
    
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
            particles_next, weekly_infections, log_likelihoods = self.propagate_and_likelihood(
                particles, Y_obs[t]
            )
            
            particles, particles_next, weekly_infections, log_likelihoods, log_Z, resample_indices = self.annealing_step(
                particles, particles_next, weekly_infections, log_likelihoods, Y_obs[t], time_step=t
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
            beta_path, infection_path = self.sample_path_with_ancestry(
                particles_history, infections_history, ancestry_matrix
            )
            
            return total_log_marginal_likelihood, beta_path, infection_path
        else:
            return total_log_marginal_likelihood
    
    def sample_path_with_ancestry(self, particles_history, infections_history, ancestry_matrix):
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
        
        # Trace lineage backwards through ancestry matrix
        current_idx = final_idx
        for t in range(T-1, -1, -1):
            # Extract values for this particle at time t
            beta_path[t] = np.exp(particles_history[t][4, current_idx])
            infection_path[t] = infections_history[t][current_idx]
            
            # Move to ancestor at previous time (if not at t=0)
            if t > 0:
                # Find which particle at time t-1 led to current_idx at time t
                # This is stored in ancestry_matrix[current_idx, t-1]
                current_idx = ancestry_matrix[current_idx, t-1]
        
        return beta_path, infection_path


# ============================================================================
# PMMH ALGORITHM
# ============================================================================

class PMMH:
    """
    Particle Marginal Metropolis-Hastings for SEIR model
    
    Estimates posterior distribution of global parameters (kappa, gamma, sigma, overdispersion)
    using SMC to compute marginal likelihood.
    """
    
    def __init__(self, Y_obs, N, num_particles=1000, target_ess_ratio=0.9):
        """
        Args:
            Y_obs: Observed weekly case counts
            N: Total population size
            num_particles: Number of SMC particles
            target_ess_ratio: Target ESS ratio for SMC annealing
        """
        self.Y_obs = Y_obs
        self.N = N
        self.num_particles = num_particles
        self.target_ess_ratio = target_ess_ratio
    
    def prior_log_pdf(self, params):
        """
        Log prior density for global parameters
        params = [kappa, gamma, sigma, overdispersion]
        
        Priors:
        - kappa ~ Beta(11.4, 2.5) - mean ≈ 0.82, from Revised Functions
        - gamma ~ Beta(12.1, 10.3) - mean ≈ 0.54, from Revised Functions
        - sigma ~ Uniform(0, 1) - flat prior, no preference (Beta(1,1) = Uniform)
        - overdispersion ~ Beta(300, 180) - mean ≈ 0.625, std ≈ 0.022, extremely concentrated
          (essentially fixed at 0.625, barely moves during MCMC)
        """
        kappa, gamma, sigma, overdispersion = params
        
        # Hard bounds - must be in (0, 1)
        if not (0 < kappa < 1 and 0 < gamma < 1 and 0 < sigma < 1 and 0 < overdispersion < 1):
            return -np.inf
        
        log_prior = 0.0
        log_prior += stats.beta(11.4, 2.5).logpdf(kappa)
        log_prior += stats.beta(12.1, 10.3).logpdf(gamma)
        log_prior += stats.beta(1, 1).logpdf(sigma)  # Beta(1,1) = Uniform(0,1)
        log_prior += stats.beta(300, 180).logpdf(overdispersion)  # Extremely concentrated, essentially fixed
        
        return log_prior
    
    def proposal(self, current_params, step_sizes):
        """
        Random walk proposal for parameters
        
        Args:
            current_params: Current parameter values [kappa, gamma, sigma, overdispersion]
            step_sizes: Standard deviations for random walk [kappa_std, gamma_std, sigma_std, overdisp_std]
        """
        proposed = current_params + np.random.normal(0, step_sizes, len(current_params))
        
        # Clip to valid ranges (0, 1)
        # proposed[0] = np.clip(proposed[0], 0.0001, 0.9999)  # kappa
        # proposed[1] = np.clip(proposed[1], 0.0001, 0.9999)  # gamma
        # proposed[2] = np.clip(proposed[2], 0.0001, 0.9999)  # sigma
        # proposed[3] = np.clip(proposed[3], 0.0001, 0.9999)  # overdispersion
        
        return proposed
    
    def run_smc(self, params, return_path=False):
        """
        Run SMC with given parameters
        
        Args:
            params: [kappa, gamma, sigma, overdispersion]
            return_path: If True, also return a sampled latent path
            
        Returns:
            log_marginal_likelihood: Estimate from SMC
            If return_path=True, also returns beta_path and infection_path
        """
        kappa, gamma, sigma, overdispersion = params
        model_params = np.array([kappa, gamma, sigma])
        
        smc = SEIRSMC(
            N=self.N,
            overdispersion=overdispersion,
            model_params=model_params,
            Y_obs_initial=self.Y_obs[0],
            target_ess_ratio=self.target_ess_ratio,
            enable_mcmc=True
        )
        
        if return_path:
            log_marginal_likelihood, beta_path, infection_path = smc.run(
                self.Y_obs, self.num_particles, return_path=True
            )
            return log_marginal_likelihood, beta_path, infection_path
        else:
            log_marginal_likelihood = smc.run(self.Y_obs, self.num_particles, return_path=False)
            return log_marginal_likelihood
    
    def run(self, num_iterations, initial_params=None, step_sizes=None, verbose=True,
            save_every=100, save_file='pmmh_chain.npz'):
        """
        Run PMMH algorithm with random walk proposals
        
        Uses simple random walk Metropolis-Hastings with fixed step sizes for each parameter.
        
        PMMH also samples latent paths: each accepted iteration includes a draw
        from the posterior distribution over beta(t) and infection(t) trajectories.
        
        Args:
            num_iterations: Number of MCMC iterations
            initial_params: Starting values [kappa, gamma, sigma, overdispersion]
            step_sizes: Proposal step sizes for [kappa, gamma, sigma, overdispersion]
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
        if initial_params is None:
            initial_params = np.array([0.9, 0.6, 0.3, 0.625])
        
        if step_sizes is None:
            # Fixed step sizes for random walk proposals
            # Increased to target acceptance rate ~0.25-0.30 (was ~0.6, now doubled)
            step_sizes = np.array([0.04, 0.04, 0.10, 0.010])  # [kappa, gamma, sigma, overdispersion]
        
        T = len(self.Y_obs)
        samples = np.zeros((num_iterations, 4))
        log_likelihoods = np.zeros(num_iterations)
        beta_paths = np.zeros((num_iterations, T))
        infection_paths = np.zeros((num_iterations, T))
        accepted = 0
        
        # Initialize chain
        current_params = initial_params.copy()
        current_log_prior = self.prior_log_pdf(current_params)
        
        if verbose:
            print("Running initial SMC to get starting log-likelihood and path...")
        current_log_lik, current_beta_path, current_infection_path = self.run_smc(
            current_params, return_path=True
        )
        current_log_posterior = current_log_prior + current_log_lik
        
        if verbose:
            print(f"Initial log-likelihood: {current_log_lik:.4f}")
            print(f"Initial log-prior: {current_log_prior:.4f}")
            print(f"Initial parameters: κ={current_params[0]:.3f}, γ={current_params[1]:.3f}, σ={current_params[2]:.3f}, ϕ={current_params[3]:.3f}")
            print(f"Initial step sizes: {step_sizes}")
            print(f"\nStarting PMMH with {num_iterations} iterations...")
            print(f"Using random walk proposals (no adaptation)")
            print(f"Saving chain every {save_every} iterations to '{save_file}'")
            print("="*70)
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # Simple random walk proposal
            proposed_params = self.proposal(current_params, step_sizes)
            proposed_log_prior = self.prior_log_pdf(proposed_params)
            
            if proposed_log_prior == -np.inf:
                # Reject immediately if outside prior support
                samples[iteration] = current_params
                log_likelihoods[iteration] = current_log_lik
                beta_paths[iteration] = current_beta_path
                infection_paths[iteration] = current_infection_path
            else:
                # Run SMC with proposed parameters - always sample a path
                proposed_log_lik, proposed_beta_path, proposed_infection_path = self.run_smc(
                    proposed_params, return_path=True
                )
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
                    accepted += 1
                
                samples[iteration] = current_params
                log_likelihoods[iteration] = current_log_lik
                beta_paths[iteration] = current_beta_path
                infection_paths[iteration] = current_infection_path
            
            # Save chain periodically
            if (iteration + 1) % save_every == 0:
                np.savez_compressed(
                    save_file,
                    samples=samples[:iteration+1],
                    log_likelihoods=log_likelihoods[:iteration+1],
                    beta_paths=beta_paths[:iteration+1],
                    infection_paths=infection_paths[:iteration+1],
                    accepted=accepted,
                    iteration=iteration+1,
                    Y_obs=self.Y_obs
                )
                if verbose:
                    print(f"  [Save] Chain saved at iteration {iteration+1}")
            
            # Print progress
            print_freq = 10 if iteration < 1000 else 100
            if verbose and (iteration + 1) % print_freq == 0:
                elapsed = time.time() - start_time
                accept_rate = accepted / (iteration + 1)
                progress = (iteration + 1) / num_iterations * 100
                eta = elapsed / (iteration + 1) * (num_iterations - iteration - 1)
                print(f"[{progress:5.1f}%] Iter {iteration+1:5d}/{num_iterations} | Accept: {accept_rate:.3f} | "
                      f"Log-lik: {current_log_lik:7.2f} | "
                      f"κ={current_params[0]:.3f} γ={current_params[1]:.3f} σ={current_params[2]:.3f} ϕ={current_params[3]:.3f} | "
                      f"ETA: {eta:.0f}s")
        
        acceptance_rate = accepted / num_iterations
        
        # Final save
        np.savez_compressed(
            save_file,
            samples=samples,
            log_likelihoods=log_likelihoods,
            beta_paths=beta_paths,
            infection_paths=infection_paths,
            accepted=accepted,
            iteration=num_iterations,
            Y_obs=self.Y_obs
        )
        
        if verbose:
            print("="*70)
            print(f"PMMH completed in {time.time() - start_time:.2f} seconds")
            print(f"Overall acceptance rate: {acceptance_rate:.3f}")
            print(f"Final chain saved to '{save_file}'")
        
        return samples, log_likelihoods, beta_paths, infection_paths, acceptance_rate


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_pmmh_results(samples, log_likelihoods, burn_in=None):
    """
    Analyze PMMH output
    
    Args:
        samples: Parameter samples (num_iterations x 4)
        log_likelihoods: Log marginal likelihoods
        burn_in: Number of initial samples to discard (default: 20%)
    """
    if burn_in is None:
        burn_in = int(len(samples) * 0.2)
    
    post_burn_samples = samples[burn_in:]
    post_burn_logliks = log_likelihoods[burn_in:]
    
    param_names = ['kappa', 'gamma', 'sigma', 'overdispersion']
    
    print("\nPosterior Summary (after burn-in):")
    print("="*60)
    for i, name in enumerate(param_names):
        mean = np.mean(post_burn_samples[:, i])
        std = np.std(post_burn_samples[:, i])
        q025, q975 = np.percentile(post_burn_samples[:, i], [2.5, 97.5])
        print(f"{name:15s}: {mean:.4f} ± {std:.4f} | 95% CI: [{q025:.4f}, {q975:.4f}]")
    
    print(f"\nLog-likelihood: {np.mean(post_burn_logliks):.2f} ± {np.std(post_burn_logliks):.2f}")
    
    # Plot traces
    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    
    for i, name in enumerate(param_names):
        # Trace plot
        axes[i, 0].plot(samples[:, i], alpha=0.7)
        axes[i, 0].axvline(burn_in, color='red', linestyle='--', label='Burn-in')
        axes[i, 0].set_ylabel(name)
        axes[i, 0].set_xlabel('Iteration')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[i, 1].hist(post_burn_samples[:, i], bins=30, density=True, alpha=0.7, edgecolor='black')
        
        # Add prior distribution as dotted line
        x_range = np.linspace(0.001, 0.999, 1000)  # Avoid boundary issues
        if i == 0:  # kappa
            prior_pdf = stats.beta(11.4, 2.5).pdf(x_range)
            axes[i, 1].plot(x_range, prior_pdf, 'r--', linewidth=2, label='Prior: Beta(11.4, 2.5)')
        elif i == 1:  # gamma
            prior_pdf = stats.beta(12.1, 10.3).pdf(x_range)
            axes[i, 1].plot(x_range, prior_pdf, 'r--', linewidth=2, label='Prior: Beta(12.1, 10.3)')
        elif i == 2:  # sigma
            prior_pdf = stats.beta(1, 1).pdf(x_range)
            axes[i, 1].plot(x_range, prior_pdf, 'r--', linewidth=2, label='Prior: Beta(1, 1)')
        elif i == 3:  # overdispersion
            prior_pdf = stats.beta(300, 180).pdf(x_range)
            axes[i, 1].plot(x_range, prior_pdf, 'r--', linewidth=2, label='Prior: Beta(300, 180)')
        
        axes[i, 1].set_xlabel(name)
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    # Log-likelihood trace
    axes[4, 0].plot(log_likelihoods, alpha=0.7)
    axes[4, 0].axvline(burn_in, color='red', linestyle='--', label='Burn-in')
    axes[4, 0].set_ylabel('Log-likelihood')
    axes[4, 0].set_xlabel('Iteration')
    axes[4, 0].legend()
    axes[4, 0].grid(True, alpha=0.3)
    
    # Log-likelihood histogram
    axes[4, 1].hist(post_burn_logliks, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[4, 1].set_xlabel('Log-likelihood')
    axes[4, 1].set_ylabel('Density')
    axes[4, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pmmh_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_paths(beta_paths, infection_paths, Y_obs, burn_in=None, num_samples=100):
    """
    Plot posterior samples of latent paths
    
    Args:
        beta_paths: Array of beta trajectories (num_iterations x T)
        infection_paths: Array of infection trajectories (num_iterations x T)
        Y_obs: Observed data
        burn_in: Burn-in period (default: 20%)
        num_samples: Number of path samples to plot (default: 100)
    """
    if burn_in is None:
        burn_in = int(len(beta_paths) * 0.2)
    
    post_burn_betas = beta_paths[burn_in:]
    post_burn_infections = infection_paths[burn_in:]
    
    # Subsample paths for plotting
    indices = np.random.choice(len(post_burn_betas), size=min(num_samples, len(post_burn_betas)), replace=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot log beta paths
    log_post_burn_betas = np.log(post_burn_betas)
    for idx in indices:
        axes[0].plot(log_post_burn_betas[idx], alpha=0.05, color='blue')
    
    # Plot posterior mean and credible intervals
    log_beta_mean = np.mean(log_post_burn_betas, axis=0)
    log_beta_lower = np.percentile(log_post_burn_betas, 2.5, axis=0)
    log_beta_upper = np.percentile(log_post_burn_betas, 97.5, axis=0)
    
    axes[0].plot(log_beta_mean, color='red', linewidth=2, label='Posterior Mean')
    axes[0].fill_between(range(len(log_beta_mean)), log_beta_lower, log_beta_upper, 
                          alpha=0.3, color='red', label='95% Credible Interval')
    axes[0].set_ylabel('log(β(t))')
    axes[0].set_xlabel('Week')
    axes[0].set_title('Posterior Samples of Log Transmission Rate log(β(t))')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot infection paths vs observations
    for idx in indices:
        axes[1].plot(post_burn_infections[idx], alpha=0.05, color='blue')
    
    infection_mean = np.mean(post_burn_infections, axis=0)
    infection_lower = np.percentile(post_burn_infections, 2.5, axis=0)
    infection_upper = np.percentile(post_burn_infections, 97.5, axis=0)
    
    axes[1].plot(infection_mean, color='red', linewidth=2, label='Posterior Mean')
    axes[1].fill_between(range(len(infection_mean)), infection_lower, infection_upper,
                          alpha=0.3, color='red', label='95% Credible Interval')
    axes[1].plot(Y_obs, 'ko', markersize=4, label='Observed Cases', alpha=0.5)
    axes[1].set_ylabel('Weekly Infections')
    axes[1].set_xlabel('Week')
    axes[1].set_title('Posterior Samples of Weekly Infections vs Observed Data')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_paths.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_from_saved_results(npz_file='pmmh_chain.npz', burn_in_fraction=0.2):
    """
    Load saved PMMH results and generate all plots without re-running the chain
    
    Args:
        npz_file: Path to the saved .npz file
        burn_in_fraction: Fraction of samples to discard as burn-in (default: 0.2 = 20%)
    """
    try:
        # Load saved results
        print(f"Loading saved PMMH results from: {npz_file}")
        data = np.load(npz_file)
        
        samples = data['samples']
        log_likelihoods = data['log_likelihoods']
        beta_paths = data['beta_paths']
        infection_paths = data['infection_paths']
        Y_obs = data['Y_obs']
        accepted = data['accepted']
        iteration = data['iteration']
        
        print(f"Loaded chain with {iteration} iterations:")
        print(f"  Parameters: {samples.shape}")
        print(f"  Beta paths: {beta_paths.shape}")
        print(f"  Infection paths: {infection_paths.shape}")
        print(f"  Acceptance rate: {accepted/iteration:.3f}")
        
        # Calculate burn-in
        burn_in = int(iteration * burn_in_fraction)
        print(f"  Using burn-in: {burn_in} iterations ({burn_in_fraction*100:.0f}%)")
        
        # Generate diagnostic plots
        print("\nGenerating diagnostic plots...")
        analyze_pmmh_results(samples, log_likelihoods, burn_in=burn_in)
        
        # Generate latent path plots
        print("Generating latent path plots...")
        plot_latent_paths(beta_paths, infection_paths, Y_obs, burn_in=burn_in, num_samples=200)
        
        print("\n" + "="*50)
        print("Plots generated successfully!")
        print("  - pmmh_diagnostics.png")
        print("  - latent_paths.png")
        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: File '{npz_file}' not found.")
        print("Make sure you have run the PMMH chain first, or provide the correct file path.")
    except KeyError as e:
        print(f"Error: Missing key {e} in the saved file.")
        print("The saved file may be from an older version or corrupted.")
    except Exception as e:
        print(f"Error loading results: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if user wants to plot from saved results
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        npz_file = sys.argv[2] if len(sys.argv) > 2 else 'pmmh_chain.npz'
        plot_from_saved_results(npz_file)
        sys.exit(0)
    
    print("PMMH for SEIR Model - COVID-19 Data")
    print("="*70)
    
    # Load COVID-19 data
    data = pd.read_feather('./covid_df.feather')
    weekly_covid_df = data.groupby([pd.Grouper(key='date', freq='W-SUN')]).agg({
        'confirmed_cases': 'sum'
    }).reset_index()
    
    pop = pd.read_feather('./INEGI_2020_State_Population.feather')
    N = pop['population'].sum()
    
    # Use first 140 weeks (indices 0 to 139)
    Y_obs = weekly_covid_df['confirmed_cases'].values[:140]
    
    print(f"\nData:")
    print(f"  Population: {N:,}")
    print(f"  Time series length: {len(Y_obs)} weeks")
    print(f"  Initial cases: {Y_obs[0]:.0f}")
    print(f"  Peak cases: {Y_obs.max():.0f}")
    
    # Create PMMH sampler
    pmmh = PMMH(Y_obs, N, num_particles=500, target_ess_ratio=0.9)
    
    # Run PMMH with good initial values
    num_iterations = 125_000
    # Start with reasonable middle values that should work
    initial_params = np.array([0.5, 0.5, 0.5, 0.5])
    
    print(f"\nInitial parameters:")
    print(f"  kappa (E→I rate): {initial_params[0]:.3f}")
    print(f"  gamma (I→R rate): {initial_params[1]:.3f}")
    print(f"  sigma (β volatility): {initial_params[2]:.3f}")
    print(f"  overdispersion: {initial_params[3]:.3f}")
    
    # Run PMMH - returns parameters, log-likelihoods, AND latent paths
    samples, log_likelihoods, beta_paths, infection_paths, acceptance_rate = pmmh.run(
        num_iterations=num_iterations,
        initial_params=initial_params,
        save_every=100,
        save_file='pmmh_chain.npz',
        verbose=True
    )
    
    print(f"\nPMMH sampled:")
    print(f"  Parameters: {samples.shape}")
    print(f"  Beta paths: {beta_paths.shape}")
    print(f"  Infection paths: {infection_paths.shape}")
    
    # Analyze results
    burn_in = int(num_iterations * 0.2)
    analyze_pmmh_results(samples, log_likelihoods, burn_in=burn_in)
    
    # Plot latent paths
    print("\nPlotting posterior samples of latent paths...")
    plot_latent_paths(beta_paths, infection_paths, Y_obs, burn_in=burn_in, num_samples=200)
    
    print("\n" + "="*70)
    print("PMMH Analysis Complete!")
    print(f"Chain saved to: pmmh_chain.npz")
    print("="*70)