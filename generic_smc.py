"""
Generic Sequential Monte Carlo with Adaptive Annealing and Resampling

This module provides a flexible framework for SMC with annealing steps.
Includes a test implementation with a Normal Hidden Markov Model.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.special import logsumexp
from scipy.optimize import newton
import scipy.stats as stats


class SMCModel(ABC):
    """Abstract base class for SMC models"""
    
    @abstractmethod
    def initial_particles(self, num_particles):
        """Generate initial particles"""
        pass
    
    @abstractmethod
    def transition(self, particles_prev, **kwargs):
        """Propagate particles from time t-1 to time t"""
        pass
    
    @abstractmethod
    def log_likelihood(self, particles, observation, **kwargs):
        """Compute log-likelihood for each particle given observation"""
        pass
    
    @abstractmethod
    def mcmc_proposal(self, particles, step_size, **kwargs):
        """Generate MCMC proposals for particles"""
        pass
    
    @abstractmethod
    def is_valid(self, particles, **kwargs):
        """Check if particles are valid"""
        pass


class AnnealingScheduler:
    """Handles annealing schedule computation using cESS"""
    
    def __init__(self, target_cess_ratio=0.99, max_annealing_steps=100, tolerance=1e-4):
        self.target_cess_ratio = target_cess_ratio
        self.max_annealing_steps = max_annealing_steps
        self.tolerance = tolerance
    
    def log_cess(self, alpha, alpha_prev, log_liks):
        """Compute log conditional ESS"""
        delta = max(0.0, float(alpha - alpha_prev))
        a = delta * log_liks

        if not np.any(np.isfinite(a)):
            return -np.inf
        if np.nanstd(log_liks) == 0.0:
            return np.log(len(log_liks))

        lse1 = logsumexp(a)
        lse2 = logsumexp(2.0 * a)
        return 2.0 * lse1 - lse2

    def d_log_cess(self, alpha, alpha_prev, log_liks):
        """Derivative of log conditional ESS"""
        delta = max(0.0, float(alpha - alpha_prev))
        a = delta * log_liks

        lse1 = logsumexp(a)
        lse2 = logsumexp(2.0 * a)

        if not np.isfinite(lse1) or not np.isfinite(lse2):
            return 0.0

        s1 = np.exp(a - lse1)
        s2 = np.exp(2.0 * a - lse2)

        E1 = np.sum(s1 * log_liks)
        E2 = np.sum(s2 * log_liks)
        return 2.0 * (E1 - E2)

    def find_next_alpha(self, alpha_prev, log_likelihoods):
        """Find next annealing parameter using Newton's method"""
        N = len(log_likelihoods)
        target_log_cess = np.log(self.target_cess_ratio * N)

        f = lambda alpha: self.log_cess(alpha, alpha_prev, log_likelihoods) - target_log_cess
        fprime = lambda alpha: self.d_log_cess(alpha, alpha_prev, log_likelihoods)
        
        alpha_init = min(alpha_prev + 0.001, 1.0)

        try:
            alpha_star = newton(func=f, x0=alpha_init, tol=self.tolerance, maxiter=10, fprime=fprime)
            alpha_star = np.clip(alpha_star, alpha_prev + 1e-4, 1.0)
        except RuntimeError:
            alpha_star = min(alpha_prev + 0.001, 1.0)

        return alpha_star


class Resampler:
    """Handles particle resampling"""
    
    @staticmethod
    def systematic_resample(weights):
        """Systematic resampling"""
        num_particles = len(weights)
        positions = (np.arange(num_particles) + np.random.uniform()) / num_particles
        return np.searchsorted(np.cumsum(weights), positions)
    
    @staticmethod
    def multinomial_resample(weights):
        """Multinomial resampling"""
        num_particles = len(weights)
        return np.random.choice(num_particles, size=num_particles, p=weights)
    
    @staticmethod
    def stratified_resample(weights):
        """Stratified resampling"""
        num_particles = len(weights)
        positions = (np.arange(num_particles) + np.random.uniform(size=num_particles)) / num_particles
        return np.searchsorted(np.cumsum(weights), positions)


class GenericSMC:
    """
    Generic Sequential Monte Carlo with Adaptive Annealing and Resampling
    """
    
    def __init__(self, model, annealing_scheduler, resampler_method='systematic'):
        """
        Args:
            model: Instance of SMCModel
            annealing_scheduler: Instance of AnnealingScheduler
            resampler_method: 'systematic', 'multinomial', or 'stratified'
        """
        self.model = model
        self.scheduler = annealing_scheduler
        self.resampler = Resampler()
        self.resampler_method = resampler_method
        
    def mcmc_step(self, particles, log_likelihoods, observation, alpha, step_size=None, **kwargs):
        """Perform MCMC jittering step"""
        num_particles = particles.shape[1] if len(particles.shape) > 1 else len(particles)
        
        # Auto-compute step size if not provided
        if step_size is None:
            if len(particles.shape) > 1:
                step_size = 0.1 * np.std(particles[-1, :])
            else:
                step_size = 0.1 * np.std(particles)
        
        # Skip MCMC if step size is effectively zero
        if step_size < 1e-10:
            return particles, log_likelihoods
        
        # Generate proposals
        proposed_particles = self.model.mcmc_proposal(particles, step_size, **kwargs)
        
        # Check validity
        valid_mask = self.model.is_valid(proposed_particles, **kwargs)
        
        # Compute proposed log-likelihoods
        proposed_ll = np.full(num_particles, -np.inf)
        
        if np.any(valid_mask):
            if len(proposed_particles.shape) > 1:
                valid_particles = proposed_particles[:, valid_mask]
            else:
                valid_particles = proposed_particles[valid_mask]
            proposed_ll[valid_mask] = self.model.log_likelihood(valid_particles, observation, **kwargs)
        
        # Metropolis-Hastings accept/reject
        log_accept = np.minimum(0.0, alpha * (proposed_ll - log_likelihoods))
        accept = np.log(np.random.rand(num_particles)) < log_accept
        
        # Update accepted particles
        if np.any(accept):
            if len(particles.shape) > 1:
                particles[:, accept] = proposed_particles[:, accept]
            else:
                particles[accept] = proposed_particles[accept]
            log_likelihoods[accept] = proposed_ll[accept]
        
        return particles, log_likelihoods
    
    def adaptive_resample(self, particles, weights, ess_threshold=0.5):
        """Adaptive resampling based on ESS"""
        ess = 1.0 / np.sum(weights**2)
        num_particles = len(weights)
        
        if ess < ess_threshold * num_particles:
            # Resample
            if self.resampler_method == 'systematic':
                indices = self.resampler.systematic_resample(weights)
            elif self.resampler_method == 'multinomial':
                indices = self.resampler.multinomial_resample(weights)
            elif self.resampler_method == 'stratified':
                indices = self.resampler.stratified_resample(weights)
            else:
                raise ValueError(f"Unknown resampler method: {self.resampler_method}")
            
            if len(particles.shape) > 1:
                particles = particles[:, indices]
            else:
                particles = particles[indices]
            weights = np.ones(num_particles) / num_particles
            
        return particles, weights, ess / num_particles
    
    def annealing_step(self, particles, log_likelihoods, observation, mcmc_step_size=None, **kwargs):
        """Perform full annealing sequence from alpha=0 to alpha=1"""
        num_particles = len(log_likelihoods)
        alpha_prev = 0.0
        log_marginal_likelihood = 0.0
        annealing_step = 0
        
        while (1.0 - alpha_prev > self.scheduler.tolerance and 
               annealing_step < self.scheduler.max_annealing_steps):
            
            # Find next alpha
            alpha_new = self.scheduler.find_next_alpha(alpha_prev, log_likelihoods)
            delta_alpha = alpha_new - alpha_prev
            
            # Update weights
            ais_log_weights = delta_alpha * log_likelihoods
            log_sum_weights = logsumexp(ais_log_weights)
            ais_norm_weights = np.exp(ais_log_weights - log_sum_weights)
            
            # Update marginal likelihood estimate
            log_marginal_likelihood += log_sum_weights - np.log(num_particles)
            
            # Adaptive resampling
            particles, ais_norm_weights, ess_ratio = self.adaptive_resample(
                particles, ais_norm_weights, ess_threshold=0.5
            )
            
            # If resampling occurred, update log-likelihoods
            if ess_ratio < 0.5:
                log_likelihoods = self.model.log_likelihood(particles, observation, **kwargs)
            
            # MCMC step
            particles, log_likelihoods = self.mcmc_step(
                particles, log_likelihoods, observation, alpha_new, mcmc_step_size, **kwargs
            )
            
            alpha_prev = alpha_new
            annealing_step += 1
        
        # Final step to alpha = 1.0 if not reached
        if alpha_prev < 1.0:
            delta_alpha = 1.0 - alpha_prev
            ais_log_weights = delta_alpha * log_likelihoods
            log_sum_weights = logsumexp(ais_log_weights)
            ais_norm_weights = np.exp(ais_log_weights - log_sum_weights)
            
            log_marginal_likelihood += log_sum_weights - np.log(num_particles)
            
            # Final adaptive resampling and MCMC
            particles, ais_norm_weights, ess_ratio = self.adaptive_resample(
                particles, ais_norm_weights, ess_threshold=0.5
            )
            
            if ess_ratio < 0.5:
                log_likelihoods = self.model.log_likelihood(particles, observation, **kwargs)
            
            particles, log_likelihoods = self.mcmc_step(
                particles, log_likelihoods, observation, 1.0, mcmc_step_size, **kwargs
            )
        
        return particles, log_likelihoods, log_marginal_likelihood
    
    def run(self, observations, num_particles, mcmc_step_size=None, return_trajectories=False, **kwargs):
        """Run full SMC with annealing for time series"""
        T = len(observations)
        
        # Initialize particles
        particles = self.model.initial_particles(num_particles)
        total_log_marginal_likelihood = 0.0
        
        # Storage for trajectories
        if return_trajectories:
            if len(particles.shape) > 1:
                particle_trajectories = np.zeros([particles.shape[0], num_particles, T])
                particle_trajectories[:, :, 0] = particles
            else:
                particle_trajectories = np.zeros([num_particles, T])
                particle_trajectories[:, 0] = particles
        
        # Time loop
        for t in range(1, T):
            # Propagate particles
            particles = self.model.transition(particles, **kwargs)
            
            # Compute initial log-likelihoods
            log_likelihoods = self.model.log_likelihood(particles, observations[t], **kwargs)
            
            # Run annealing step
            particles, log_likelihoods, log_marg_lik = self.annealing_step(
                particles, log_likelihoods, observations[t], mcmc_step_size, **kwargs
            )
            
            total_log_marginal_likelihood += log_marg_lik
            
            # Store trajectories
            if return_trajectories:
                if len(particles.shape) > 1:
                    particle_trajectories[:, :, t] = particles
                else:
                    particle_trajectories[:, t] = particles
        
        if return_trajectories:
            return total_log_marginal_likelihood, particles, particle_trajectories
        else:
            return total_log_marginal_likelihood, particles


# ============================================================================
# TEST CASE: Normal Hidden Markov Model
# ============================================================================

class NormalHMM(SMCModel):
    """
    Simple Normal Hidden Markov Model for testing
    
    Hidden state: X_t ~ N(phi * X_{t-1}, sigma_x^2)
    Observation: Y_t ~ N(X_t, sigma_y^2)
    
    This has an analytic solution via Kalman filter for comparison.
    """
    
    def __init__(self, phi=0.9, sigma_x=1.0, sigma_y=0.5, x0_mean=0.0, x0_var=1.0):
        self.phi = phi
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.x0_mean = x0_mean
        self.x0_var = x0_var
    
    def initial_particles(self, num_particles):
        """Generate initial particles from prior"""
        return np.random.normal(self.x0_mean, np.sqrt(self.x0_var), num_particles)
    
    def transition(self, particles_prev, **kwargs):
        """Transition: X_t ~ N(phi * X_{t-1}, sigma_x^2)"""
        noise = np.random.normal(0, self.sigma_x, len(particles_prev))
        return self.phi * particles_prev + noise
    
    def log_likelihood(self, particles, observation, **kwargs):
        """Likelihood: Y_t ~ N(X_t, sigma_y^2)"""
        return stats.norm(particles, self.sigma_y).logpdf(observation)
    
    def mcmc_proposal(self, particles, step_size, **kwargs):
        """Simple random walk proposal"""
        return particles + np.random.normal(0, step_size, len(particles))
    
    def is_valid(self, particles, **kwargs):
        """All particles are valid for this model"""
        return np.ones(len(particles), dtype=bool)
    
    def kalman_filter(self, observations):
        """
        Analytic solution using Kalman filter
        Returns log marginal likelihood and filtered means/variances
        """
        T = len(observations)
        
        # Initialize
        m = self.x0_mean
        P = self.x0_var
        log_likelihood = 0.0
        
        means = np.zeros(T)
        variances = np.zeros(T)
        means[0] = m
        variances[0] = P
        
        for t in range(1, T):
            # Predict
            m_pred = self.phi * m
            P_pred = self.phi**2 * P + self.sigma_x**2
            
            # Update
            S = P_pred + self.sigma_y**2  # Innovation variance
            K = P_pred / S  # Kalman gain
            
            innovation = observations[t] - m_pred
            m = m_pred + K * innovation
            P = P_pred - K * P_pred
            
            # Log likelihood contribution
            log_likelihood += stats.norm(m_pred, np.sqrt(S)).logpdf(observations[t])
            
            means[t] = m
            variances[t] = P
        
        return log_likelihood, means, variances


def test_normal_hmm():
    """Test the generic SMC framework with Normal HMM"""
    
    print("Testing Generic SMC with Normal Hidden Markov Model")
    print("=" * 60)
    
    # Model parameters
    phi = 0.9
    sigma_x = 1.0
    sigma_y = 0.5
    T = 50
    num_particles = 1000
    
    # Create model
    hmm = NormalHMM(phi=phi, sigma_x=sigma_x, sigma_y=sigma_y)
    
    # Generate synthetic data
    np.random.seed(42)
    true_states = np.zeros(T)
    observations = np.zeros(T)
    
    true_states[0] = np.random.normal(0, 1)
    observations[0] = np.random.normal(true_states[0], sigma_y)
    
    for t in range(1, T):
        true_states[t] = phi * true_states[t-1] + np.random.normal(0, sigma_x)
        observations[t] = np.random.normal(true_states[t], sigma_y)
    
    # Run Kalman filter (analytic solution)
    kf_log_lik, kf_means, kf_vars = hmm.kalman_filter(observations)
    
    # Run SMC
    scheduler = AnnealingScheduler(target_cess_ratio=0.9, max_annealing_steps=50)
    smc = GenericSMC(hmm, scheduler, 'systematic')
    
    np.random.seed(42)  # Same seed for fair comparison
    smc_log_lik, final_particles, trajectories = smc.run(
        observations, num_particles, mcmc_step_size=0.1, return_trajectories=True
    )
    
    # Compute SMC estimates
    smc_means = np.mean(trajectories, axis=0)
    smc_vars = np.var(trajectories, axis=0)
    
    # Compare results
    print(f"Analytic log-likelihood (Kalman):  {kf_log_lik:.4f}")
    print(f"SMC log-likelihood:                {smc_log_lik:.4f}")
    print(f"Difference:                        {abs(kf_log_lik - smc_log_lik):.4f}")
    print()
    
    # RMSE comparison
    state_rmse = np.sqrt(np.mean((smc_means - kf_means)**2))
    var_rmse = np.sqrt(np.mean((smc_vars - kf_vars)**2))
    
    print(f"State estimation RMSE:             {state_rmse:.4f}")
    print(f"Variance estimation RMSE:          {var_rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: States
    plt.subplot(1, 3, 1)
    plt.plot(true_states, 'k-', label='True States', linewidth=2)
    plt.plot(observations, 'ro', alpha=0.5, markersize=3, label='Observations')
    plt.plot(kf_means, 'b--', label='Kalman Filter', linewidth=2)
    plt.plot(smc_means, 'r:', label='SMC', linewidth=2)
    plt.fill_between(range(T), 
                     kf_means - 1.96*np.sqrt(kf_vars), 
                     kf_means + 1.96*np.sqrt(kf_vars), 
                     alpha=0.2, color='blue', label='95% CI (Kalman)')
    plt.legend()
    plt.title('State Estimation')
    plt.xlabel('Time')
    plt.ylabel('State Value')
    
    # Plot 2: Particle evolution for a few time points
    plt.subplot(1, 3, 2)
    for t in [10, 20, 30, 40]:
        plt.hist(trajectories[:, t], bins=30, alpha=0.3, density=True, label=f't={t}')
    plt.title('Particle Distributions Over Time')
    plt.xlabel('State Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 3: Log-likelihood comparison
    plt.subplot(1, 3, 3)
    plt.bar(['Kalman Filter', 'SMC'], [kf_log_lik, smc_log_lik], 
            color=['blue', 'red'], alpha=0.7)
    plt.title('Log Marginal Likelihood')
    plt.ylabel('Log Likelihood')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'kf_log_lik': kf_log_lik,
        'smc_log_lik': smc_log_lik,
        'state_rmse': state_rmse,
        'var_rmse': var_rmse,
        'true_states': true_states,
        'observations': observations,
        'kf_means': kf_means,
        'smc_means': smc_means,
        'trajectories': trajectories
    }


if __name__ == "__main__":
    results = test_normal_hmm()