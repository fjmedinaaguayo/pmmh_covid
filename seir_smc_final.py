"""
SEIR-Specific SMC Implementation

This version is specifically designed for the SEIR model to handle
the fact that we need to call transition to get weekly infections
before computing likelihoods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import logsumexp


# ============================================================================
# YOUR EXISTING BM_SEIR FUNCTION (unchanged)
# ============================================================================

def BM_SEIR(V_in, params, num_particles, N, m):
    """Your existing BM_SEIR function - no changes needed"""
    num_steps = m + 1  
    h = 1 / num_steps 

    # Initialize arrays to store results
    V = np.zeros([V_in.shape[0], num_particles, num_steps + 1])    
    new_infected = np.zeros([num_particles])

    # Set initial conditions
    kappa, gamma, sigma = params[:3]    
    V[:,:,0] = V_in

    # Simulate the SEIR model
    for t in range(1, num_steps + 1):
        # Clip log_beta to reasonable bounds - matches notebook
        log_beta_capped = np.clip(V[4,:,t-1], -1, 1)  # exp(-1) ≈ 0.37, exp(1) ≈ 2.72
        
        infections = np.exp(log_beta_capped) * V[0, :, t-1] * V[2, :, t-1] / N
        latent = kappa * V[1, :, t-1]
        recovered = gamma * V[2, :, t-1]
        
        V[0, :, t] = V[0, :, t-1] - infections*h                  # S
        V[1, :, t] = V[1, :, t-1] + (infections  - latent)*h      # E
        V[2, :, t] = V[2, :, t-1] + (latent - recovered)*h        # I
        V[3, :, t] = V[3, :, t-1] + recovered*h                   # R
        
        # Ensure non-negative populations
        V[0, :, t] = np.maximum(V[0, :, t], 0)
        V[1, :, t] = np.maximum(V[1, :, t], 0)
        V[2, :, t] = np.maximum(V[2, :, t], 0)
        V[3, :, t] = np.maximum(V[3, :, t], 0)
        
        # Update latent state
        dB = stats.norm(0, 1).rvs(num_particles)
        V[4,:,t] = V[4,:,t-1] + sigma * np.sqrt(h) * dB
        
        # Update weekly infected count
        new_infected += infections*h
    
    return V[:,:,-1], new_infected


# ============================================================================
# SEIR-SPECIFIC SMC CLASS
# ============================================================================

class SEIRSMC:
    """
    SMC implementation specifically designed for SEIR model
    
    This handles the SEIR model's specific requirements:
    - Particles are [S, E, I, R, log_beta] states
    - Weekly infections are computed via BM_SEIR
    - Likelihood uses negative binomial distribution
    """
    
    def __init__(self, N, overdispersion, model_params, Y_obs_initial, 
                 target_ess_ratio=0.9, enable_mcmc=True): 
        """
        Args:
            N: Population size
            overdispersion: Overdispersion parameter
            model_params: [kappa, gamma, sigma]
            Y_obs_initial: Y_obs[0] for initialization
            target_ess_ratio: Target ESS ratio for annealing
            enable_mcmc: Whether to use MCMC jittering
        """
        self.N = N
        self.overdispersion = overdispersion
        self.model_params = model_params
        self.Y_obs_initial = Y_obs_initial
        self.target_ess_ratio = target_ess_ratio
        self.enable_mcmc = enable_mcmc
    
    def initial_particles(self, num_particles):
        """Generate initial SEIR particles matching notebook initialization"""
        I0 = np.ones(num_particles) * self.Y_obs_initial
        E0 = np.random.uniform(0, 0.0001 * self.N, num_particles)  # Match notebook: 0 to 0.01% of population
        R0 = np.zeros(num_particles)  # No recovered initially
        S0 = self.N - E0 - I0 - R0
        
        # Initial log_beta from uniform(0, 2) like notebook
        log_beta0 = np.log(np.random.uniform(0.01, 2, num_particles))  # Avoid log(0)
        
        return np.array([S0, E0, I0, R0, log_beta0])
    
    def propagate_and_likelihood(self, particles, observation, m=0):
        """
        Propagate particles and compute likelihoods
        
        Returns:
            particles_next: Propagated particles
            weekly_infections: Weekly infections for likelihood
            log_likelihoods: Log-likelihoods for each particle
        """
        num_particles = particles.shape[1]
        
        # Propagate using BM_SEIR
        particles_next, weekly_infections = BM_SEIR(
            particles, self.model_params, num_particles, self.N, m
        )
        
        # Compute log-likelihoods with correct negative binomial parameterization
        log_likelihoods = np.full(num_particles, -np.inf)
        
        # Only compute likelihood for valid particles
        valid_mask = (weekly_infections > 0) & np.isfinite(weekly_infections)
        
        if np.any(valid_mask):
            # Use correct negative binomial parameterization from notebook
            # Mean = n*(1-p)/p, we want mean = weekly_infections
            # With overdispersion φ: mean = μ, variance = μ + φ*μ²
            # This gives: n = 1/φ, p = 1/(1 + φ*μ)
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
        """MCMC jittering step for log_beta only - VECTORIZED"""
        if not self.enable_mcmc:
            if return_accept_rate:
                return particles_t, log_likelihoods, 0.0
            else:
                return particles_t, log_likelihoods
        
        num_particles = particles_t.shape[1]
        
        # Vectorized proposal
        proposed_log_beta = particles_t[-1, :] + np.random.normal(0, step_size, num_particles)
        proposed_log_beta = np.clip(proposed_log_beta, -1, 1)  # Match BM_SEIR bounds
        
        # Create proposed particles
        proposed_particles = particles_t.copy()
        proposed_particles[-1, :] = proposed_log_beta
        
        # Vectorized propagation
        proposed_next, proposed_infections = BM_SEIR(
            proposed_particles, self.model_params, num_particles, self.N, 0
        )
        
        # Vectorized validity check
        valid_mask = (proposed_infections > 0) & np.isfinite(proposed_infections) & np.all(proposed_next[:4, :] >= 0, axis=0)
        
        # Vectorized likelihood computation
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
        
        # Update accepted particles
        particles_t[-1, accept] = proposed_log_beta[accept]
        log_likelihoods[accept] = proposed_ll[accept]
        
        accept_rate = np.sum(accept) / num_particles
        if return_accept_rate:
            return particles_t, log_likelihoods, accept_rate
        else:
            return particles_t, log_likelihoods
    
    def find_next_alpha(self, alpha_prev, log_liks, tolerance=0.01, verbose=False):
        """Find next annealing parameter using conditional ESS with robust bisection
        
        Args:
            tolerance: Relative tolerance for cESS ratio (e.g., 0.01 means 1% tolerance)
        """
        def compute_cess(alpha):
            """Compute conditional ESS for given alpha value"""
            if alpha <= alpha_prev:
                return len(log_liks)  # If no progress, ESS = N
            
            # Incremental weight exponent: (alpha - alpha_prev) * log_liks
            delta = alpha - alpha_prev
            a = delta * log_liks
            
            # Handle case where all log_liks are -inf
            finite_mask = np.isfinite(a)
            if not np.any(finite_mask):
                return 0.0  # No finite particles
            
            # If all log_liks are the same, ESS = N
            finite_liks = log_liks[finite_mask]
            if len(finite_liks) > 1 and np.var(finite_liks) < 1e-10:
                return len(log_liks)
            
            # Compute ESS stably using the standard formula
            # ESS = (sum exp(a))^2 / sum exp(2a)
            max_a = np.max(a[finite_mask])
            a_shifted = a[finite_mask] - max_a
            
            exp_a = np.exp(a_shifted)
            exp_2a = np.exp(2 * a_shifted)
            
            sum_exp_a = np.sum(exp_a)
            sum_exp_2a = np.sum(exp_2a)
            
            if sum_exp_a <= 0 or sum_exp_2a <= 0:
                return 0.0
            
            # ESS = (sum w)^2 / sum w^2
            ess = (sum_exp_a ** 2) / sum_exp_2a
            
            return ess
        
        target_ess = self.target_ess_ratio * len(log_liks)
        
        # Store diagnostics
        current_ess = compute_cess(alpha_prev)
        
        # Check if we can reach target at alpha=1
        ess_at_one = compute_cess(1.0)
        
        if verbose:
            print(f"    Initial: α_prev={alpha_prev:.4f}, cESS={current_ess:.1f}")
            print(f"    Target: cESS={target_ess:.1f}")
            print(f"    At α=1.0: cESS={ess_at_one:.1f}")
        
        if ess_at_one > target_ess:
            # Can't reach target even at alpha=1 (cESS still too high), use adaptive strategy
            alpha_target = min(alpha_prev + 0.1, 1.0)  # Take 10% step toward alpha=1
            alpha_final = alpha_target
            converged = False
            if verbose:
                print(f"    Cannot reach target cESS (still {ess_at_one:.1f} > {target_ess:.1f} at α=1), taking conservative step to α={alpha_final:.4f}")
        else:
            # Bisection search (we can reach the target)
            alpha_low = alpha_prev
            alpha_high = 1.0
            converged = False
            
            for iteration in range(50):  # Max 50 iterations with ratio-based tolerance
                alpha_mid = (alpha_low + alpha_high) / 2
                ess_mid = compute_cess(alpha_mid)
                
                if verbose:
                    ess_ratio = ess_mid / len(log_liks)
                    target_ratio = target_ess / len(log_liks)
                    print(f"      Iter {iteration+1}: α_mid={alpha_mid:.6f}, cESS={ess_mid:.1f} (ratio={ess_ratio:.3f}), target={target_ess:.1f} (ratio={target_ratio:.3f})")
                
                # Check convergence using relative tolerance on the ratio
                ess_ratio = ess_mid / len(log_liks)
                target_ratio = self.target_ess_ratio
                if abs(ess_ratio - target_ratio) < tolerance:
                    alpha_final = alpha_mid
                    converged = True
                    if verbose:
                        print(f"      Converged at iteration {iteration+1}: ratio difference {abs(ess_ratio - target_ratio):.4f} < {tolerance}")
                    break
                
                if ess_mid > target_ess:
                    alpha_low = alpha_mid  # ESS too high, need higher alpha (search upper half)
                    if verbose:
                        print(f"      cESS too high, searching upper half: [{alpha_mid:.6f}, {alpha_high:.6f}]")
                else:
                    alpha_high = alpha_mid  # ESS too low, need lower alpha (search lower half)
                    if verbose:
                        print(f"      cESS too low, searching lower half: [{alpha_low:.6f}, {alpha_mid:.6f}]")
                
                if abs(alpha_high - alpha_low) < 1e-6:
                    if verbose:
                        print(f"      Bisection interval too small, stopping")
                    break
            
            alpha_final = (alpha_low + alpha_high) / 2
        
        # Ensure we make progress
        alpha_final = max(alpha_final, alpha_prev + 1e-6)
        alpha_final = min(alpha_final, 1.0)
        
        final_ess = compute_cess(alpha_final)
        
        if verbose:
            print(f"    Bisection: α={alpha_prev:.4f} → {alpha_final:.4f}, "
                  f"cESS={current_ess:.1f} → {final_ess:.1f} (target={target_ess:.1f}), "
                  f"converged={converged}")
        
        return alpha_final
    
    def systematic_resample(self, weights):
        """Systematic resampling"""
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform()) / n
        return np.searchsorted(np.cumsum(weights), positions)
    
    def annealing_step(self, particles_t, particles_tplus1, weekly_infections, 
                       log_likelihoods, observation, time_step=None):
        """Full annealing sequence from alpha=0 to alpha=1"""
        alpha = 0.0
        log_Z = 0.0
        max_steps = 100
        
        # Initialize cumulative log-weights (for tracking actual ESS)
        cumulative_log_weights = np.zeros(len(log_likelihoods))
        
        # Remove invalid particles before annealing
        valid_mask = self.is_valid(particles_tplus1, weekly_infections)
        if not np.any(valid_mask):
            print("Warning: No valid particles!")
            return particles_t, particles_tplus1, weekly_infections, log_likelihoods, -np.inf
        
        # Set invalid log-likelihoods to -inf
        log_likelihoods[~valid_mask] = -np.inf
        
        for step in range(max_steps):
            if 1.0 - alpha < 1e-4:
                break
            
            # Find next alpha using conditional ESS (without verbose diagnostics)
            alpha_new = self.find_next_alpha(alpha, log_likelihoods, verbose=False)
            delta = alpha_new - alpha
            
            # Update incremental weights for log evidence
            incremental_log_weights = delta * log_likelihoods
            
            # Check for numerical issues
            if not np.any(np.isfinite(incremental_log_weights)):
                print("Warning: All weights are -inf, stopping annealing")
                break
                
            # Add to log evidence using incremental weights
            log_Z += logsumexp(incremental_log_weights) - np.log(len(incremental_log_weights))
            
            # Update cumulative log-weights: w(α) = exp(α * log_lik)
            cumulative_log_weights = alpha_new * log_likelihoods
            
            # Normalize cumulative weights for ESS computation and resampling
            weights = np.exp(cumulative_log_weights - logsumexp(cumulative_log_weights))
            
            # Check ESS based on cumulative weights
            ess = 1.0 / np.sum(weights**2)
            
            if ess < len(weights) * 0.5:  # Resample if ESS < N/2
                indices = self.systematic_resample(weights)
                particles_t = particles_t[:, indices]
                particles_tplus1 = particles_tplus1[:, indices]
                weekly_infections = weekly_infections[indices]
                log_likelihoods = log_likelihoods[indices]
                # Reset cumulative weights after resampling (all particles now have equal weight)
                cumulative_log_weights = np.zeros(len(log_likelihoods))
            
            # MCMC step (only jitter log_beta at time t)
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
        
        # Final propagation after annealing completes to get final particles_tplus1
        particles_tplus1, weekly_infections, log_likelihoods = self.propagate_and_likelihood(
            particles_t, observation
        )
        
        return particles_t, particles_tplus1, weekly_infections, log_likelihoods, log_Z
    
    def run(self, Y_obs, num_particles, return_trajectories=False):
        """Run SEIR SMC for full time series"""
        T = len(Y_obs)
        
        # Initialize
        particles = self.initial_particles(num_particles)
        total_log_marginal_likelihood = 0.0
        
        # Storage for trajectories
        if return_trajectories:
            particle_trajectories = np.zeros([5, num_particles, T])
            weekly_infections_trajectories = np.zeros([num_particles, T])
            particle_trajectories[:, :, 0] = particles
            # For t=0, weekly infections = I0
            weekly_infections_trajectories[:, 0] = particles[2, :]
        
        # Time loop
        for t in range(1, T):
            if t % 10 == 0:
                print(f"Processing time step {t}/{T-1}...")
            
            # Propagate and compute likelihoods
            particles_next, weekly_infections, log_likelihoods = self.propagate_and_likelihood(
                particles, Y_obs[t]
            )
            
            # Annealing step (minimal diagnostics)
            particles, particles_next, weekly_infections, log_likelihoods, log_Z = self.annealing_step(
                particles, particles_next, weekly_infections, log_likelihoods, Y_obs[t], time_step=None
            )
            
            # Update for next iteration
            particles = particles_next
            total_log_marginal_likelihood += log_Z
            
            if t % 10 == 0:
                mean_inf = np.mean(weekly_infections)
                mean_beta = np.mean(np.exp(particles_next[-1, :]))
                print(f"  Cumulative log marginal likelihood: {total_log_marginal_likelihood:.4f}")
                print(f"  Mean weekly infections: {mean_inf:.0f}, Observed: {Y_obs[t]:.0f}")
                print(f"  Mean β: {mean_beta:.4f}, Mean I: {np.mean(particles_next[2,:]):.0f}")
            
            # Store trajectories
            if return_trajectories:
                particle_trajectories[:, :, t] = particles
                weekly_infections_trajectories[:, t] = weekly_infections
        
        if return_trajectories:
            return total_log_marginal_likelihood, particles, particle_trajectories, weekly_infections_trajectories
        else:
            return total_log_marginal_likelihood, particles


# ============================================================================
# COVID-19 DATA ANALYSIS
# ============================================================================

def load_covid_data():
    """Load and preprocess COVID-19 data"""
    try:
        # Load COVID data
        data = pd.read_feather('./covid_df.feather')
        
        # Create weekly aggregation
        weekly_covid_df = data.groupby([pd.Grouper(key='date', freq='W-SUN')]).agg({
            'confirmed_cases': 'sum',
            'confirmed_deaths': 'sum'
        }).reset_index()
        
        # Load population data
        pop = pd.read_feather('./INEGI_2020_State_Population.feather')
        N = pop['population'].sum()
        
        # Get observations (first 140 weeks)
        Y_obs = weekly_covid_df['confirmed_cases'].values[:140]
        
        print(f"Loaded COVID-19 data:")
        print(f"  Total population: {N:,}")
        print(f"  Weekly observations: {len(Y_obs)}")
        print(f"  Date range: {len(Y_obs)} weeks")
        print(f"  Peak cases: {np.max(Y_obs):,}")
        print(f"  Initial cases: {Y_obs[0]:,}")
        
        return Y_obs, N, weekly_covid_df
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure covid_df.feather and INEGI_2020_State_Population.feather are in the current directory")
        return None, None, None

def run_covid_analysis():
    """Run SEIR SMC analysis on COVID-19 data"""
    
    print("SEIR SMC Analysis of COVID-19 Data")
    print("=" * 50)
    
    # Load COVID-19 data
    Y_obs, N, weekly_covid_df = load_covid_data()
    if Y_obs is None:
        return None
    
    # Model parameters (from your notebook)
    overdispersion = 0.6212418740345191
    model_params = np.array([0.9, 0.6, 0.3])  # [kappa, gamma, sigma]
    num_particles = 1000  # Use 1000 particles as in your original code
    
    # Use all 140 timesteps for full analysis  
    T_analysis = len(Y_obs)
    Y_obs_short = Y_obs[:T_analysis]    
    print(f"Analysis configuration:")
    print(f"  Population size: {N:,}")
    print(f"  Time series length: {T_analysis} weeks")
    print(f"  Number of particles: {num_particles}")
    print(f"  Model parameters: kappa={model_params[0]}, gamma={model_params[1]}, sigma={model_params[2]}")
    print(f"  Overdispersion: {overdispersion}")
    print(f"  Initial observed cases: {Y_obs_short[0]:,}")
    print()
    
    # Create and run SMC
    print("Running SEIR SMC on COVID-19 data...")
    smc = SEIRSMC(
        N=N,
        overdispersion=overdispersion,
        model_params=model_params,
        Y_obs_initial=Y_obs_short[0],
        target_ess_ratio=0.9,
        enable_mcmc=True
    )
    
    import time
    start_time = time.time()
    log_lik, final_particles, trajectories, weekly_infections = smc.run(
        Y_obs_short, num_particles, return_trajectories=True
    )
    runtime = time.time() - start_time
    
    print(f"Analysis completed in {runtime:.2f} seconds")
    print(f"Log marginal likelihood: {log_lik:.4f}")
    print(f"Final particle statistics:")
    print(f"  S: {np.mean(final_particles[0, :]):.0f} ± {np.std(final_particles[0, :]):.0f}")
    print(f"  E: {np.mean(final_particles[1, :]):.0f} ± {np.std(final_particles[1, :]):.0f}")
    print(f"  I: {np.mean(final_particles[2, :]):.0f} ± {np.std(final_particles[2, :]):.0f}")
    print(f"  R: {np.mean(final_particles[3, :]):.0f} ± {np.std(final_particles[3, :]):.0f}")
    print(f"  log_beta: {np.mean(final_particles[4, :]):.3f} ± {np.std(final_particles[4, :]):.3f}")
    
    # Plot results
    plt.figure(figsize=(16, 8))
    
    # Plot 1: Compartment evolution
    plt.subplot(2, 2, 1)
    compartment_names = ['S', 'E', 'I', 'R']
    colors = ['blue', 'orange', 'red', 'green']
    
    for i in range(4):
        mean_traj = np.mean(trajectories[i, :, :], axis=0)
        std_traj = np.std(trajectories[i, :, :], axis=0)
        plt.plot(mean_traj, color=colors[i], label=compartment_names[i], linewidth=2)
        plt.fill_between(range(T_analysis), mean_traj - std_traj, mean_traj + std_traj, 
                        alpha=0.3, color=colors[i])
    
    plt.title('SEIR Compartment Evolution')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Population')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: log_beta evolution
    plt.subplot(2, 2, 2)
    log_beta_mean = np.mean(trajectories[4, :, :], axis=0)
    log_beta_std = np.std(trajectories[4, :, :], axis=0)
    plt.plot(log_beta_mean, 'purple', linewidth=2, label='log β')
    plt.fill_between(range(T_analysis), log_beta_mean - log_beta_std, log_beta_mean + log_beta_std, 
                    alpha=0.3, color='purple')
    plt.title('Transmission Rate Evolution')
    plt.xlabel('Time (weeks)')
    plt.ylabel('log β')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Weekly infections vs observations
    plt.subplot(2, 2, 3)
    weekly_mean = np.mean(weekly_infections, axis=0)
    weekly_std = np.std(weekly_infections, axis=0)
    
    plt.plot(Y_obs_short, 'ro-', label='Observed Cases', markersize=4, linewidth=2)
    plt.plot(weekly_mean, 'b-', label='Model Weekly Infections', linewidth=2)
    plt.fill_between(range(T_analysis), weekly_mean - weekly_std, weekly_mean + weekly_std, 
                    alpha=0.3, color='blue')
    plt.title('Model Fit: Weekly Infections vs Observations')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Cases')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Particle trajectories (sample)
    plt.subplot(2, 2, 4)
    sample_particles = min(50, num_particles)
    for i in range(sample_particles):
        plt.plot(weekly_infections[i, :], color='gray', alpha=0.1, linewidth=0.5)
    plt.plot(Y_obs_short, 'ro-', label='Observed', markersize=3, linewidth=2)
    plt.plot(weekly_mean, 'b-', label='Mean', linewidth=2)
    plt.title(f'Particle Trajectories (sample of {sample_particles})')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Weekly Infections')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print(f"\nModel fit statistics:")
    mse = np.mean((Y_obs_short - weekly_mean)**2)
    mae = np.mean(np.abs(Y_obs_short - weekly_mean))
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  Mean Absolute Error: {mae:.2f}")
    
    # Correlation between observed and predicted
    correlation = np.corrcoef(Y_obs_short, weekly_mean)[0, 1]
    print(f"  Correlation: {correlation:.4f}")
    
    return log_lik, final_particles, trajectories, weekly_infections, Y_obs_short


if __name__ == "__main__":
    results = run_covid_analysis()
    
    if results is not None:
        print("\n" + "="*60)
        print("SUCCESS: SEIR SMC Analysis Complete!")
        print("="*60)
        print("\nKey features successfully applied to COVID-19 data:")
        print("✅ Loaded real COVID-19 weekly case data")
        print("✅ Used actual Mexican population size")
        print("✅ Applied adaptive annealing with conditional ESS")
        print("✅ Implemented systematic resampling")
        print("✅ Used MCMC jittering for particle diversity")
        print("✅ Generated comprehensive model fit diagnostics")
        print("✅ Integrated with your existing BM_SEIR function")
        print("✅ Handled 1000 particles efficiently")
        print("\nThe SMC framework has been successfully applied to your COVID-19 data!")
        print("Model provides log-likelihood for Bayesian model comparison.")
    else:
        print("❌ Could not load COVID-19 data files.")
        print("Please ensure covid_df.feather and INEGI_2020_State_Population.feather are available.")


