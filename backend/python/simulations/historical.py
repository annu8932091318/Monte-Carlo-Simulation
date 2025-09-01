"""
Historical Simulation Module

This module implements historical simulation methods for risk assessment.
Historical simulation uses actual past returns to generate potential future scenarios,
providing a non-parametric approach to risk modeling that captures tail risks and
non-normal distributions present in real market data.

Key Features:
- Historical simulation with rolling windows
- Bootstrap resampling of historical data
- Non-parametric risk estimation
- Actual market stress scenarios
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings
from .math_helpers import (
    generate_normal_random,
    calculate_portfolio_statistics,
    calculate_correlation_matrix,
    calculate_risk_metrics
)

def generate_historical_returns(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Generate historical returns data.
    
    Since we don't have actual historical data, we'll generate synthetic
    historical returns that exhibit realistic market characteristics including:
    - Volatility clustering
    - Fat tails
    - Occasional extreme events
    
    Args:
        portfolio: Portfolio asset values
        params: Parameters including lookback_periods, volatility patterns
        
    Returns:
        Array of historical returns [periods, assets]
    """
    if params is None:
        params = {}
    
    num_assets = len(portfolio)
    lookback_periods = params.get('lookback_periods', 252)  # 1 year daily data
    base_volatilities = params.get('volatilities', [0.15] * num_assets)
    base_correlations = params.get('correlations', 0.3)
    
    # Create synthetic historical returns with realistic market features
    returns = np.zeros((lookback_periods, num_assets))
    
    # Generate correlation matrix
    if isinstance(base_correlations, (int, float)):
        correlation_matrix = np.full((num_assets, num_assets), base_correlations)
        np.fill_diagonal(correlation_matrix, 1.0)
    else:
        correlation_matrix = np.array(base_correlations)
    
    # Cholesky decomposition for correlated returns
    try:
        chol_matrix = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If correlation matrix is not positive definite, use identity
        warnings.warn("Correlation matrix not positive definite, using identity")
        chol_matrix = np.eye(num_assets)
    
    # Generate returns with different market regimes
    regime_changes = [0, 60, 120, 180, 240]  # Regime change points
    
    for i, start_period in enumerate(regime_changes):
        end_period = regime_changes[i + 1] if i + 1 < len(regime_changes) else lookback_periods
        period_length = end_period - start_period
        
        # Different market regimes
        if i % 3 == 0:  # Normal market
            vol_multiplier = 1.0
            skew_factor = 0.0
        elif i % 3 == 1:  # High volatility market
            vol_multiplier = 1.8
            skew_factor = -0.5  # Negative skew
        else:  # Low volatility market
            vol_multiplier = 0.6
            skew_factor = 0.3  # Positive skew
        
        # Generate independent normal returns
        independent_returns = np.random.standard_normal((period_length, num_assets))
        
        # Add skewness to simulate realistic return distributions
        if skew_factor != 0:
            skewed_component = np.random.exponential(1, (period_length, num_assets))
            skewed_component = (skewed_component - np.mean(skewed_component)) / np.std(skewed_component)
            independent_returns += skew_factor * skewed_component
        
        # Apply correlation structure
        correlated_returns = independent_returns @ chol_matrix.T
        
        # Scale by volatilities and regime multiplier
        for j in range(num_assets):
            correlated_returns[:, j] *= base_volatilities[j] * vol_multiplier
        
        returns[start_period:end_period] = correlated_returns
    
    # Add occasional extreme events (market crashes/rallies)
    num_extreme_events = max(1, lookback_periods // 100)
    extreme_periods = np.random.choice(lookback_periods, num_extreme_events, replace=False)
    
    for period in extreme_periods:
        # Random extreme event affecting all assets
        extreme_magnitude = np.random.uniform(-0.15, 0.10)  # -15% to +10%
        market_shock = np.random.uniform(0.7, 1.0, num_assets)  # 70% to 100% correlation
        returns[period] += extreme_magnitude * market_shock
    
    return returns

def historical_simulation(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Historical simulation using past return data.
    
    This method uses historical returns to simulate future portfolio values
    by randomly sampling from the historical distribution. It preserves the
    actual distribution characteristics including fat tails and asymmetry.
    
    Args:
        portfolio: List of asset values
        params: Simulation parameters
        
    Returns:
        Dictionary with simulation results and risk metrics
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    iterations = params.get('iterations', 100000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)  # Time horizon in years
    lookback_periods = params.get('lookback_periods', 252)
    
    portfolio = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio)
    
    print(f"🏛️ Running Historical Simulation:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Lookback Periods: {lookback_periods}")
    print(f"  Time Horizon: {horizon} years")
    
    # Generate historical returns
    historical_returns = generate_historical_returns(portfolio, params)
    
    # Convert to daily returns if horizon is different from 1 day
    # Assuming historical_returns are daily returns
    if horizon != 1/252:  # If not 1 day
        # Scale returns for the time horizon
        time_scaling = np.sqrt(horizon * 252)  # Assuming 252 trading days per year
        scaled_returns = historical_returns * time_scaling
    else:
        scaled_returns = historical_returns
    
    # Historical simulation: randomly sample returns with replacement
    simulation_results = np.zeros(iterations)
    
    # Calculate portfolio weights
    weights = portfolio / initial_value
    
    for i in range(iterations):
        # Randomly sample one period from historical returns
        sample_period = np.random.randint(0, len(scaled_returns))
        period_returns = scaled_returns[sample_period]
        
        # Calculate portfolio return
        portfolio_return = np.sum(weights * period_returns)
        
        # Calculate terminal portfolio value
        terminal_value = initial_value * (1 + portfolio_return)
        simulation_results[i] = terminal_value
    
    # Calculate risk metrics
    losses = initial_value - simulation_results
    profit_loss = simulation_results - initial_value
    
    risk_metrics = calculate_risk_metrics(
        simulation_results, 
        initial_value, 
        confidence
    )
    
    # Additional historical simulation specific metrics
    returns_distribution = (simulation_results / initial_value) - 1
    
    # Historical simulation statistics
    worst_case = np.min(simulation_results)
    best_case = np.max(simulation_results)
    median_case = np.median(simulation_results)
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(simulation_results, percentiles)
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'historical_simulation',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio.tolist(),
            'iterations': iterations,
            'confidence': confidence,
            'horizon_years': horizon,
            'lookback_periods': lookback_periods,
            'historical_approach': 'non_parametric_resampling'
        },
        
        'risk_metrics': risk_metrics,
        
        'historical_analysis': {
            'worst_case_scenario': float(worst_case),
            'best_case_scenario': float(best_case),
            'median_scenario': float(median_case),
            'scenario_range': float(best_case - worst_case),
            'downside_scenarios': float(np.sum(simulation_results < initial_value) / iterations * 100),
            'extreme_loss_scenarios': float(np.sum(losses > risk_metrics['VaR_95']) / iterations * 100)
        },
        
        'distribution_analysis': {
            'return_percentiles': {
                f'p{p}': float(val / initial_value - 1) * 100
                for p, val in zip(percentiles, percentile_values)
            },
            'skewness': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 3)),
            'kurtosis': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 4)) - 3,
            'tail_ratio': float(np.mean(returns_distribution[returns_distribution < np.percentile(returns_distribution, 5)]) / 
                              np.mean(returns_distribution[returns_distribution > np.percentile(returns_distribution, 95)])) * -1
        },
        
        'portfolio_stats': {
            'expected_terminal': float(np.mean(simulation_results)),
            'expected_return': float(np.mean(returns_distribution) * 100),
            'volatility': float(np.std(returns_distribution) * 100),
            'sharpe_ratio': float(np.mean(returns_distribution) / np.std(returns_distribution)) if np.std(returns_distribution) > 0 else 0,
            'historical_volatility': float(np.std(historical_returns.flatten()) * np.sqrt(252) * 100)
        },
        
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time) if simulation_time > 0 else 0,
            'historical_periods_analyzed': int(lookback_periods),
            'total_scenarios_evaluated': int(iterations)
        }
    }

def bootstrap_historical(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Bootstrap historical simulation with overlapping periods.
    
    This method enhances historical simulation by using bootstrap resampling
    to create synthetic historical scenarios, allowing for better coverage
    of potential outcomes while preserving the empirical distribution.
    
    Args:
        portfolio: List of asset values
        params: Simulation parameters
        
    Returns:
        Dictionary with bootstrap simulation results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Extract parameters
    iterations = params.get('iterations', 100000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    lookback_periods = params.get('lookback_periods', 252)
    block_size = params.get('block_size', 5)  # Block bootstrap size
    
    portfolio = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio)
    
    print(f"🔄 Running Bootstrap Historical Simulation:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Block Size: {block_size}")
    print(f"  Time Horizon: {horizon} years")
    
    # Generate historical returns
    historical_returns = generate_historical_returns(portfolio, params)
    
    # Calculate portfolio weights
    weights = portfolio / initial_value
    
    # Convert historical returns to portfolio returns
    historical_portfolio_returns = np.sum(historical_returns * weights[np.newaxis, :], axis=1)
    
    # Bootstrap simulation results
    simulation_results = np.zeros(iterations)
    
    # Block bootstrap to preserve some temporal structure
    num_blocks = len(historical_portfolio_returns) - block_size + 1
    
    for i in range(iterations):
        # Construct a bootstrap path
        path_length = max(1, int(horizon * 252))  # Number of periods for the horizon
        bootstrap_returns = np.zeros(path_length)
        
        for j in range(0, path_length, block_size):
            # Randomly select a block starting point
            block_start = np.random.randint(0, num_blocks)
            block_end = min(block_start + block_size, len(historical_portfolio_returns))
            block_returns = historical_portfolio_returns[block_start:block_end]
            
            # Fill the bootstrap path
            end_idx = min(j + len(block_returns), path_length)
            bootstrap_returns[j:end_idx] = block_returns[:end_idx - j]
        
        # Calculate cumulative return
        cumulative_return = np.prod(1 + bootstrap_returns) - 1
        terminal_value = initial_value * (1 + cumulative_return)
        simulation_results[i] = terminal_value
    
    # Calculate risk metrics and analysis
    risk_metrics = calculate_risk_metrics(
        simulation_results, 
        initial_value, 
        confidence
    )
    
    # Bootstrap-specific analysis
    returns_distribution = (simulation_results / initial_value) - 1
    
    # Stability analysis - compare with simple historical simulation
    simple_hist_results = np.zeros(min(10000, iterations))
    for i in range(len(simple_hist_results)):
        sample_return = np.random.choice(historical_portfolio_returns)
        simple_hist_results[i] = initial_value * (1 + sample_return * horizon)
    
    simple_var = np.percentile(initial_value - simple_hist_results, confidence * 100)
    bootstrap_var = risk_metrics['VaR_95']
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'bootstrap_historical',
            'engine': 'python', 
            'initial_value': float(initial_value),
            'portfolio': portfolio.tolist(),
            'iterations': iterations,
            'confidence': confidence,
            'horizon_years': horizon,
            'lookback_periods': lookback_periods,
            'block_size': block_size,
            'bootstrap_approach': 'block_bootstrap_with_overlap'
        },
        
        'risk_metrics': risk_metrics,
        
        'bootstrap_analysis': {
            'bootstrap_var_stability': float(abs(bootstrap_var - simple_var) / simple_var * 100),
            'effective_scenarios': int(len(np.unique(simulation_results.round(2)))),
            'block_coverage': float(block_size / lookback_periods * 100),
            'temporal_correlation_preserved': True if block_size > 1 else False
        },
        
        'distribution_analysis': {
            'bootstrap_skewness': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 3)),
            'bootstrap_kurtosis': float(np.mean(((returns_distribution - np.mean(returns_distribution)) / np.std(returns_distribution)) ** 4)) - 3,
            'empirical_coverage': float(np.sum(simulation_results >= np.percentile(simulation_results, (1-confidence)*100)) / len(simulation_results) * 100),
            'distribution_span': float(np.max(returns_distribution) - np.min(returns_distribution))
        },
        
        'portfolio_stats': {
            'expected_terminal': float(np.mean(simulation_results)),
            'expected_return': float(np.mean(returns_distribution) * 100),
            'volatility': float(np.std(returns_distribution) * 100),
            'sharpe_ratio': float(np.mean(returns_distribution) / np.std(returns_distribution)) if np.std(returns_distribution) > 0 else 0,
            'bootstrap_efficiency': float(len(np.unique(simulation_results)) / iterations * 100)
        },
        
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time) if simulation_time > 0 else 0,
            'bootstrap_blocks_generated': int(iterations * horizon * 252 / block_size),
            'historical_data_utilization': float(100.0)  # Bootstrap uses all historical data
        }
    }

# Export functions
__all__ = [
    'historical_simulation',
    'bootstrap_historical',
    'generate_historical_returns'
]
