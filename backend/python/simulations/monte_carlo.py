"""
Monte Carlo Risk Simulation using Geometric Brownian Motion - Python Implementation

This module implements Monte Carlo simulation for portfolio risk assessment
using the Geometric Brownian Motion model. It generates thousands of possible
future portfolio values and calculates risk metrics.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
try:
    from utils.math_helpers import (
        calculate_var, calculate_cvar, calculate_statistics,
        calculate_portfolio_stats, correlation_to_covariance,
        generate_correlated_normals, validate_portfolio,
        validate_correlation_matrix
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.math_helpers import (
        calculate_var, calculate_cvar, calculate_statistics,
        calculate_portfolio_stats, correlation_to_covariance,
        generate_correlated_normals, validate_portfolio,
        validate_correlation_matrix
    )

def monte_carlo_simulation(portfolio: List[float], 
                         params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Monte Carlo simulation for portfolio risk using GBM
    
    Mathematical Foundation:
    Portfolio value follows: S(T) = S(0) * exp((μ - σ²/2) * T + σ * √T * Z)
    Where Z ~ N(0,1) is standard normal random variable
    
    Args:
        portfolio: List of portfolio asset values
        params: Simulation parameters
    
    Returns:
        Dictionary with risk metrics and simulation results
    """
    if params is None:
        params = {}
    
    # Default parameters
    iterations = params.get('iterations', 1000000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    
    # Validate inputs first
    portfolio_array = validate_portfolio(portfolio)
    n_assets = len(portfolio_array)
    
    # Dynamic default parameters based on portfolio size
    if n_assets == 3:
        expected_returns = np.array(params.get('expected_returns', [0.08, 0.05, 0.12]))
        volatilities = np.array(params.get('volatilities', [0.20, 0.15, 0.25]))
        correlation_matrix = np.array(params.get('correlation_matrix', [
            [1.00, 0.40, 0.30],
            [0.40, 1.00, 0.35],
            [0.30, 0.35, 1.00]
        ]))
    elif n_assets == 4:
        expected_returns = np.array(params.get('expected_returns', [0.08, 0.05, 0.12, 0.03]))
        volatilities = np.array(params.get('volatilities', [0.20, 0.15, 0.25, 0.10]))
        correlation_matrix = np.array(params.get('correlation_matrix', [
            [1.00, 0.40, 0.30, 0.10],
            [0.40, 1.00, 0.35, 0.15],
            [0.30, 0.35, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00]
        ]))
    else:
        # General case: create defaults for any number of assets
        expected_returns = np.array(params.get('expected_returns', 
                                             [0.08] * n_assets))  # Default 8% return for all
        volatilities = np.array(params.get('volatilities', 
                                          [0.20] * n_assets))   # Default 20% volatility for all
        # Identity matrix with some correlation
        correlation_matrix = np.array(params.get('correlation_matrix',
                                                np.eye(n_assets) * 0.7 + np.ones((n_assets, n_assets)) * 0.3))
    
    correlation_matrix = validate_correlation_matrix(correlation_matrix)
    
    # Calculate initial portfolio value and weights
    initial_value = np.sum(portfolio_array)
    weights = portfolio_array / initial_value
    
    # Convert correlation to covariance matrix
    covariance_matrix = correlation_to_covariance(volatilities, correlation_matrix)
    
    # Calculate portfolio-level parameters
    portfolio_stats = calculate_portfolio_stats(weights, expected_returns, covariance_matrix)
    mu_p = portfolio_stats['expected_return']
    sigma_p = portfolio_stats['volatility']
    
    print(f"🎲 Starting Monte Carlo simulation with {iterations:,} iterations...")
    start_time = time.time()
    
    # Pre-calculate GBM parameters
    drift_term = (mu_p - 0.5 * sigma_p * sigma_p) * horizon
    diffusion_std = sigma_p * np.sqrt(horizon)
    
    # Generate random shocks and calculate terminal values
    random_shocks = np.random.standard_normal(iterations)
    terminal_values = initial_value * np.exp(drift_term + diffusion_std * random_shocks)
    
    simulation_time = time.time() - start_time
    
    # Calculate risk metrics
    losses = initial_value - terminal_values
    var_95 = calculate_var(losses, confidence)
    cvar_95 = calculate_cvar(losses, confidence)
    
    # Calculate statistics
    terminal_stats = calculate_statistics(terminal_values)
    
    # Calculate probability of loss
    probability_of_loss = np.mean(terminal_values < initial_value)
    
    # Calculate percentiles
    p10 = np.percentile(terminal_values, 10)
    p50 = np.percentile(terminal_values, 50)
    p90 = np.percentile(terminal_values, 90)
    
    return {
        # Simulation Configuration
        'simulation_config': {
            'method': 'monte_carlo',
            'engine': 'python',
            'iterations': iterations,
            'horizon': horizon,
            'confidence': confidence,
            'initial_value': float(initial_value),
            'portfolio_mu': float(mu_p),
            'portfolio_sigma': float(sigma_p)
        },
        
        # Risk Metrics
        'risk_metrics': {
            'VaR_95': float(var_95),
            'CVaR_95': float(cvar_95),
            'probability_of_loss': float(probability_of_loss * 100),
            'expected_shortfall': float(cvar_95)
        },
        
        # Portfolio Statistics
        'portfolio_stats': {
            'expected_terminal': terminal_stats['mean'],
            'standard_deviation': terminal_stats['std'],
            'expected_return': float((terminal_stats['mean'] / initial_value - 1) * 100),
            'volatility': float(sigma_p * 100)
        },
        
        # Scenario Analysis
        'scenarios': {
            'pessimistic_p10': float(p10),
            'median_p50': float(p50),
            'optimistic_p90': float(p90)
        },
        
        # Performance Metrics
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time)
        },
        
        # Distribution data (subset for plotting)
        'distribution': terminal_values[:min(10000, iterations)].tolist()
    }

def advanced_monte_carlo_simulation(portfolio: List[float], 
                                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Advanced Monte Carlo with correlation modeling
    
    This version explicitly models asset correlations and generates
    correlated return paths for each asset individually.
    
    Args:
        portfolio: List of portfolio asset values
        params: Advanced simulation parameters
    
    Returns:
        Dictionary with detailed risk metrics and asset-level analysis
    """
    if params is None:
        params = {}
    
    # Default parameters
    iterations = params.get('iterations', 1000000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    expected_returns = np.array(params.get('expected_returns', [0.08, 0.05, 0.12, 0.03]))
    volatilities = np.array(params.get('volatilities', [0.20, 0.15, 0.25, 0.10]))
    correlation_matrix = np.array(params.get('correlation_matrix', [
        [1.00, 0.40, 0.30, 0.10],
        [0.40, 1.00, 0.35, 0.15],
        [0.30, 0.35, 1.00, 0.05],
        [0.10, 0.15, 0.05, 1.00]
    ]))
    
    # Validate inputs
    portfolio_array = validate_portfolio(portfolio)
    correlation_matrix = validate_correlation_matrix(correlation_matrix)
    
    num_assets = len(portfolio_array)
    initial_value = np.sum(portfolio_array)
    
    print(f"🎯 Starting Advanced Monte Carlo simulation with {iterations:,} iterations...")
    start_time = time.time()
    
    # Convert correlation to covariance matrix for random generation
    cov_matrix_for_generation = correlation_matrix.copy()
    
    # Generate correlated random returns for all assets
    correlated_returns = generate_correlated_normals(iterations, cov_matrix_for_generation)
    
    # Initialize arrays for asset paths
    asset_terminal_values = np.zeros((iterations, num_assets))
    terminal_values = np.zeros(iterations)
    
    # Simulate each asset individually
    for i in range(iterations):
        for j in range(num_assets):
            # Apply GBM to each asset individually
            drift_term = (expected_returns[j] - 0.5 * volatilities[j] ** 2) * horizon
            diffusion_term = volatilities[j] * np.sqrt(horizon) * correlated_returns[i, j]
            
            asset_terminal_value = portfolio_array[j] * np.exp(drift_term + diffusion_term)
            asset_terminal_values[i, j] = asset_terminal_value
        
        terminal_values[i] = np.sum(asset_terminal_values[i, :])
    
    simulation_time = time.time() - start_time
    
    # Calculate portfolio-level risk metrics
    losses = initial_value - terminal_values
    var_95 = calculate_var(losses, confidence)
    cvar_95 = calculate_cvar(losses, confidence)
    
    # Calculate asset-level statistics
    asset_analysis = []
    for j in range(num_assets):
        asset_stats = calculate_statistics(asset_terminal_values[:, j])
        asset_losses = portfolio_array[j] - asset_terminal_values[:, j]
        asset_var = calculate_var(asset_losses, confidence)
        
        asset_analysis.append({
            'asset_index': j,
            'initial_value': float(portfolio_array[j]),
            'expected_terminal': asset_stats['mean'],
            'standard_deviation': asset_stats['std'],
            'expected_return': float((asset_stats['mean'] / portfolio_array[j] - 1) * 100),
            'volatility': float(volatilities[j] * 100),
            'VaR_95': float(asset_var)
        })
    
    # Get basic monte carlo results for comparison
    basic_results = monte_carlo_simulation(portfolio, params)
    
    return {
        **basic_results,  # Include all basic results
        
        # Advanced asset-level analysis
        'asset_analysis': asset_analysis,
        
        # Enhanced performance metrics
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time),
            'assets_simulated': num_assets,
            'total_computations': iterations * num_assets
        }
    }
