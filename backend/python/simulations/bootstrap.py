"""
Bootstrap Simulation Module

This module implements various bootstrap methods for risk simulation.
Bootstrap methods provide robust non-parametric alternatives to Monte Carlo
simulation by resampling from empirical distributions or synthetic data.

Key Features:
- Standard bootstrap resampling
- Block bootstrap for time series
- Parametric bootstrap
- Confidence interval estimation
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings

def calculate_basic_stats(data: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistical measures"""
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data))
    }

def calculate_risk_metrics_local(
    simulation_results: np.ndarray,
    initial_value: float,
    confidence: float = 0.95
) -> Dict[str, float]:
    """Calculate risk metrics locally"""
    losses = initial_value - simulation_results
    profits = simulation_results - initial_value
    
    # Value at Risk (VaR)
    var_percentile = confidence * 100
    var_95 = np.percentile(losses, var_percentile)
    
    # Conditional Value at Risk (CVaR)
    var_threshold_losses = losses[losses >= var_95]
    cvar_95 = np.mean(var_threshold_losses) if len(var_threshold_losses) > 0 else var_95
    
    # Probability of loss
    prob_loss = np.sum(simulation_results < initial_value) / len(simulation_results) * 100
    
    # Expected shortfall
    shortfall_scenarios = simulation_results[simulation_results < initial_value]
    expected_shortfall = np.mean(initial_value - shortfall_scenarios) if len(shortfall_scenarios) > 0 else 0
    
    return {
        'VaR_95': float(var_95),
        'CVaR_95': float(cvar_95),
        'probability_of_loss': float(prob_loss),
        'expected_shortfall': float(expected_shortfall),
        'maximum_loss': float(np.max(losses)),
        'maximum_gain': float(np.max(profits))
    }

def bootstrap_simulation(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standard bootstrap simulation for portfolio risk assessment.
    
    This method uses bootstrap resampling to generate scenarios from
    the empirical distribution of returns, providing robust non-parametric
    risk estimates that don't rely on distributional assumptions.
    
    Args:
        portfolio: List of asset values
        params: Simulation parameters
        
    Returns:
        Dictionary with bootstrap simulation results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    iterations = params.get('iterations', 100000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    sample_size = params.get('sample_size', 252)  # Historical sample size
    
    portfolio_array = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio_array)
    num_assets = len(portfolio_array)
    weights = portfolio_array / initial_value
    
    print(f"🥾 Running Bootstrap Simulation:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Sample Size: {sample_size}")
    print(f"  Time Horizon: {horizon} years")
    
    # Generate synthetic historical returns for bootstrap sampling
    # This simulates what would be historical return data
    base_returns = generate_synthetic_returns(num_assets, sample_size, params)
    
    # Convert to portfolio returns
    portfolio_returns = np.sum(base_returns * weights[np.newaxis, :], axis=1)
    
    # Bootstrap simulation
    simulation_results = np.zeros(iterations)
    bootstrap_samples = np.zeros((iterations, sample_size))
    
    for i in range(iterations):
        # Bootstrap resample with replacement
        bootstrap_indices = np.random.choice(sample_size, sample_size, replace=True)
        bootstrap_sample = portfolio_returns[bootstrap_indices]
        bootstrap_samples[i] = bootstrap_sample
        
        # Calculate scenario for the time horizon
        if horizon <= 1/252:  # Single period
            scenario_return = np.random.choice(bootstrap_sample)
        else:
            # Multi-period: compound returns
            periods = int(horizon * 252)
            period_returns = np.random.choice(bootstrap_sample, periods, replace=True)
            scenario_return = np.prod(1 + period_returns) - 1
        
        terminal_value = initial_value * (1 + scenario_return)
        simulation_results[i] = terminal_value
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics_local(simulation_results, initial_value, confidence)
    
    # Bootstrap-specific analysis
    returns_distribution = (simulation_results / initial_value) - 1
    
    # Bootstrap distribution properties
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    bootstrap_stds = np.std(bootstrap_samples, axis=1)
    
    # Confidence intervals for statistics
    mean_ci_lower = np.percentile(bootstrap_means, (1 - confidence) * 50)
    mean_ci_upper = np.percentile(bootstrap_means, (1 + confidence) * 50)
    
    std_ci_lower = np.percentile(bootstrap_stds, (1 - confidence) * 50)
    std_ci_upper = np.percentile(bootstrap_stds, (1 + confidence) * 50)
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'bootstrap',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio_array.tolist(),
            'iterations': iterations,
            'confidence': confidence,
            'horizon_years': horizon,
            'sample_size': sample_size,
            'bootstrap_type': 'standard_nonparametric'
        },
        
        'risk_metrics': risk_metrics,
        
        'bootstrap_analysis': {
            'mean_confidence_interval': [float(mean_ci_lower), float(mean_ci_upper)],
            'volatility_confidence_interval': [float(std_ci_lower), float(std_ci_upper)],
            'bootstrap_bias': float(np.mean(bootstrap_means) - np.mean(portfolio_returns)),
            'bootstrap_variance': float(np.var(bootstrap_means)),
            'effective_sample_diversity': float(len(np.unique(simulation_results.round(4))) / iterations * 100)
        },
        
        'distribution_analysis': {
            'empirical_skewness': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 3)),
            'empirical_kurtosis': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 4)) - 3,
            'distribution_support': [float(np.min(returns_distribution)), float(np.max(returns_distribution))],
            'tail_characteristics': {
                'left_tail_weight': float(np.sum(returns_distribution < np.percentile(returns_distribution, 5)) / len(returns_distribution) * 100),
                'right_tail_weight': float(np.sum(returns_distribution > np.percentile(returns_distribution, 95)) / len(returns_distribution) * 100)
            }
        },
        
        'portfolio_stats': {
            'expected_terminal': float(np.mean(simulation_results)),
            'expected_return': float(np.mean(returns_distribution) * 100),
            'volatility': float(np.std(returns_distribution) * 100),
            'sharpe_ratio': float(np.mean(returns_distribution) / np.std(returns_distribution)) if np.std(returns_distribution) > 0 else 0,
            'bootstrap_stability': float(np.std(bootstrap_means) / np.mean(bootstrap_means) * 100) if np.mean(bootstrap_means) != 0 else 0
        },
        
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time) if simulation_time > 0 else 0,
            'bootstrap_efficiency': float(sample_size * iterations / simulation_time) if simulation_time > 0 else 0,
            'resampling_operations': int(iterations * sample_size)
        }
    }

def advanced_bootstrap(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Advanced bootstrap simulation with multiple resampling strategies.
    
    This method implements sophisticated bootstrap techniques including:
    - Stratified bootstrap
    - Balanced bootstrap
    - Smooth bootstrap
    - Block bootstrap for time series
    
    Args:
        portfolio: List of asset values
        params: Advanced simulation parameters
        
    Returns:
        Dictionary with advanced bootstrap results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Extract parameters
    iterations = params.get('iterations', 100000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    sample_size = params.get('sample_size', 252)
    bootstrap_method = params.get('bootstrap_method', 'stratified')
    block_size = params.get('block_size', 10)
    
    portfolio_array = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio_array)
    num_assets = len(portfolio_array)
    weights = portfolio_array / initial_value
    
    print(f"🚀 Running Advanced Bootstrap Simulation:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Method: {bootstrap_method}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Block Size: {block_size}")
    
    # Generate base data
    base_returns = generate_synthetic_returns(num_assets, sample_size, params)
    portfolio_returns = np.sum(base_returns * weights[np.newaxis, :], axis=1)
    
    # Advanced bootstrap simulation
    simulation_results = np.zeros(iterations)
    method_specific_stats = {}
    
    if bootstrap_method == 'stratified':
        # Stratified bootstrap: ensure representation from different market conditions
        # Divide returns into quantiles and sample from each
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        quantile_boundaries = np.percentile(portfolio_returns, [q * 100 for q in quantiles])
        
        for i in range(iterations):
            stratified_sample = []
            samples_per_stratum = sample_size // (len(quantiles) - 1)
            
            for j in range(len(quantiles) - 1):
                # Find returns in this stratum
                stratum_mask = (portfolio_returns >= quantile_boundaries[j]) & (portfolio_returns < quantile_boundaries[j + 1])
                stratum_returns = portfolio_returns[stratum_mask]
                
                if len(stratum_returns) > 0:
                    # Sample from this stratum
                    stratum_sample = np.random.choice(stratum_returns, 
                                                    min(samples_per_stratum, len(stratum_returns)), 
                                                    replace=True)
                    stratified_sample.extend(stratum_sample)
            
            # Fill remaining samples if needed
            while len(stratified_sample) < sample_size:
                stratified_sample.append(np.random.choice(portfolio_returns))
            
            stratified_sample = np.array(stratified_sample[:sample_size])
            
            # Calculate scenario
            if horizon <= 1/252:
                scenario_return = np.random.choice(stratified_sample)
            else:
                periods = int(horizon * 252)
                period_returns = np.random.choice(stratified_sample, periods, replace=True)
                scenario_return = np.prod(1 + period_returns) - 1
            
            simulation_results[i] = initial_value * (1 + scenario_return)
        
        method_specific_stats = {
            'stratification_quantiles': len(quantiles) - 1,
            'stratum_representation': 'equal_weighted'
        }
    
    elif bootstrap_method == 'block':
        # Block bootstrap: preserve temporal correlation
        num_blocks = sample_size - block_size + 1
        
        for i in range(iterations):
            # Generate path using block bootstrap
            path_length = max(1, int(horizon * 252))
            bootstrap_path = []
            
            while len(bootstrap_path) < path_length:
                # Randomly select a block
                block_start = np.random.randint(0, num_blocks)
                block_end = min(block_start + block_size, len(portfolio_returns))
                block = portfolio_returns[block_start:block_end]
                
                # Add block to path
                remaining_needed = path_length - len(bootstrap_path)
                bootstrap_path.extend(block[:remaining_needed])
            
            bootstrap_path = np.array(bootstrap_path[:path_length])
            
            # Calculate cumulative return
            if len(bootstrap_path) == 1:
                scenario_return = bootstrap_path[0]
            else:
                scenario_return = np.prod(1 + bootstrap_path) - 1
            
            simulation_results[i] = initial_value * (1 + scenario_return)
        
        method_specific_stats = {
            'block_size': block_size,
            'blocks_available': num_blocks,
            'temporal_correlation_preserved': True
        }
    
    elif bootstrap_method == 'smooth':
        # Smooth bootstrap: add noise to bootstrap samples
        noise_factor = params.get('noise_factor', 0.1)
        
        for i in range(iterations):
            # Standard bootstrap sample
            bootstrap_sample = np.random.choice(portfolio_returns, sample_size, replace=True)
            
            # Add smooth noise
            noise = np.random.normal(0, np.std(bootstrap_sample) * noise_factor, sample_size)
            smooth_sample = bootstrap_sample + noise
            
            # Calculate scenario
            if horizon <= 1/252:
                scenario_return = np.random.choice(smooth_sample)
            else:
                periods = int(horizon * 252)
                period_returns = np.random.choice(smooth_sample, periods, replace=True)
                scenario_return = np.prod(1 + period_returns) - 1
            
            simulation_results[i] = initial_value * (1 + scenario_return)
        
        method_specific_stats = {
            'noise_factor': noise_factor,
            'smoothing_applied': True,
            'kernel_type': 'gaussian'
        }
    
    else:  # balanced bootstrap
        # Balanced bootstrap: ensure each observation appears equal number of times
        appearances_per_obs = iterations // sample_size
        remainder = iterations % sample_size
        
        # Create balanced index array
        balanced_indices = []
        for obs_idx in range(sample_size):
            balanced_indices.extend([obs_idx] * appearances_per_obs)
        
        # Add remainder randomly
        if remainder > 0:
            balanced_indices.extend(np.random.choice(sample_size, remainder, replace=False))
        
        np.random.shuffle(balanced_indices)
        
        for i in range(iterations):
            obs_idx = balanced_indices[i]
            scenario_return = portfolio_returns[obs_idx]
            
            if horizon > 1/252:
                # For multi-period, use compound effect
                periods = int(horizon * 252)
                scenario_return = (1 + scenario_return) ** (horizon * 252) - 1
            
            simulation_results[i] = initial_value * (1 + scenario_return)
        
        method_specific_stats = {
            'appearances_per_observation': appearances_per_obs,
            'perfect_balance': remainder == 0,
            'remainder_observations': remainder
        }
    
    # Calculate comprehensive metrics
    risk_metrics = calculate_risk_metrics_local(simulation_results, initial_value, confidence)
    returns_distribution = (simulation_results / initial_value) - 1
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'advanced_bootstrap',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio_array.tolist(),
            'iterations': iterations,
            'confidence': confidence,
            'horizon_years': horizon,
            'sample_size': sample_size,
            'bootstrap_method': bootstrap_method,
            'advanced_features': method_specific_stats
        },
        
        'risk_metrics': risk_metrics,
        
        'advanced_analysis': {
            'method_efficiency': float(len(np.unique(simulation_results.round(4))) / iterations * 100),
            'distribution_stability': float(np.std(simulation_results) / np.mean(simulation_results) * 100),
            'convergence_quality': float(abs(np.mean(returns_distribution) - np.mean((portfolio_returns - 1))) * 100),
            **method_specific_stats
        },
        
        'distribution_analysis': {
            'advanced_skewness': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 3)),
            'advanced_kurtosis': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 4)) - 3,
            'tail_behavior': {
                'extreme_loss_frequency': float(np.sum(returns_distribution < -0.1) / len(returns_distribution) * 100),
                'extreme_gain_frequency': float(np.sum(returns_distribution > 0.1) / len(returns_distribution) * 100)
            }
        },
        
        'portfolio_stats': {
            'expected_terminal': float(np.mean(simulation_results)),
            'expected_return': float(np.mean(returns_distribution) * 100),
            'volatility': float(np.std(returns_distribution) * 100),
            'sharpe_ratio': float(np.mean(returns_distribution) / np.std(returns_distribution)) if np.std(returns_distribution) > 0 else 0,
            'advanced_stability': float(np.std(simulation_results[-1000:]) / np.std(simulation_results[:1000])) if iterations >= 2000 else 1.0
        },
        
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time) if simulation_time > 0 else 0,
            'method_overhead': float(simulation_time / iterations * 1000) if iterations > 0 else 0,  # ms per iteration
            'bootstrap_complexity': bootstrap_method
        }
    }

def generate_synthetic_returns(
    num_assets: int,
    sample_size: int,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """Generate synthetic return data for bootstrap sampling"""
    if params is None:
        params = {}
    
    # Default parameters
    base_returns = params.get('expected_returns', [0.08] * num_assets)
    volatilities = params.get('volatilities', [0.15] * num_assets)
    correlations = params.get('correlations', 0.3)
    
    # Create correlation matrix
    if isinstance(correlations, (int, float)):
        correlation_matrix = np.full((num_assets, num_assets), correlations)
        np.fill_diagonal(correlation_matrix, 1.0)
    else:
        correlation_matrix = np.array(correlations)
    
    # Generate correlated returns
    try:
        chol_matrix = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        chol_matrix = np.eye(num_assets)
    
    # Generate returns with some realistic features
    returns = np.zeros((sample_size, num_assets))
    
    for i in range(sample_size):
        # Base random returns
        independent_returns = np.random.standard_normal(num_assets)
        
        # Apply correlation
        correlated_returns = chol_matrix @ independent_returns
        
        # Scale by volatilities and add expected returns
        for j in range(num_assets):
            returns[i, j] = base_returns[j] / 252 + volatilities[j] / np.sqrt(252) * correlated_returns[j]
    
    return returns

# Export functions
__all__ = [
    'bootstrap_simulation',
    'advanced_bootstrap',
    'generate_synthetic_returns'
]
