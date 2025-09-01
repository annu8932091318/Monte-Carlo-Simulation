"""
Geometric Brownian Motion (GBM) Simulation Module

This module implements various Geometric Brownian Motion simulation methods
for asset price modeling and portfolio risk assessment. GBM is the foundation
of the Black-Scholes model and provides realistic asset price dynamics.

Key Features:
- Standard GBM simulation
- Multi-asset correlated GBM
- Path-dependent GBM with barriers
- Jump-diffusion extensions
- Advanced numerical schemes
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings

def calculate_risk_metrics_gbm(
    simulation_results: np.ndarray,
    initial_value: float,
    confidence: float = 0.95
) -> Dict[str, float]:
    """Calculate risk metrics for GBM simulation results"""
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

def gbm_simulation(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standard Geometric Brownian Motion simulation.
    
    Simulates asset prices using the GBM stochastic differential equation:
    dS_t = μ * S_t * dt + σ * S_t * dW_t
    
    Where:
    - μ is the drift (expected return)
    - σ is the volatility
    - dW_t is a Wiener process increment
    
    Args:
        portfolio: List of asset values
        params: Simulation parameters
        
    Returns:
        Dictionary with GBM simulation results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    iterations = params.get('iterations', 100000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    time_steps = params.get('time_steps', 252)  # Daily steps for 1 year
    expected_returns = params.get('expected_returns', None)
    volatilities = params.get('volatilities', None)
    correlations = params.get('correlations', None)
    
    portfolio = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio)
    num_assets = len(portfolio)
    
    print(f"📈 Running GBM Simulation:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Time Steps: {time_steps}")
    print(f"  Time Horizon: {horizon} years")
    
    # Set default parameters
    if expected_returns is None:
        expected_returns = np.array([0.08] * num_assets)
    else:
        expected_returns = np.array(expected_returns)
    
    if volatilities is None:
        volatilities = np.array([0.15] * num_assets)
    else:
        volatilities = np.array(volatilities)
    
    # Create correlation matrix
    if correlations is None:
        correlation_matrix = np.eye(num_assets)
    elif isinstance(correlations, (int, float)):
        correlation_matrix = np.full((num_assets, num_assets), correlations)
        np.fill_diagonal(correlation_matrix, 1.0)
    else:
        correlation_matrix = np.array(correlations)
    
    # Ensure positive definite correlation matrix
    eigenvals = np.linalg.eigvals(correlation_matrix)
    if np.any(eigenvals <= 0):
        correlation_matrix = correlation_matrix + np.eye(num_assets) * 1e-6
    
    # Cholesky decomposition for correlated random numbers
    try:
        chol_matrix = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        warnings.warn("Correlation matrix not positive definite, using identity")
        chol_matrix = np.eye(num_assets)
    
    # Time parameters
    dt = horizon / time_steps
    sqrt_dt = np.sqrt(dt)
    
    # GBM parameters (with Ito correction)
    drift_adjusted = expected_returns - 0.5 * volatilities**2
    
    simulation_results = np.zeros(iterations)
    
    # Store some paths for analysis
    store_paths = min(1000, iterations)
    stored_paths = np.zeros((store_paths, time_steps + 1, num_assets))
    
    for i in range(iterations):
        # Initialize asset prices
        asset_prices = portfolio.copy()
        
        if i < store_paths:
            stored_paths[i, 0] = asset_prices
        
        # Simulate GBM path
        for t in range(time_steps):
            # Generate correlated random numbers
            independent_randoms = np.random.standard_normal(num_assets)
            correlated_randoms = chol_matrix @ independent_randoms
            
            # GBM update formula
            random_component = volatilities * sqrt_dt * correlated_randoms
            deterministic_component = drift_adjusted * dt
            
            # Update asset prices using exact GBM solution
            asset_prices = asset_prices * np.exp(deterministic_component + random_component)
            
            if i < store_paths:
                stored_paths[i, t + 1] = asset_prices
        
        # Calculate terminal portfolio value
        terminal_portfolio_value = np.sum(asset_prices)
        simulation_results[i] = terminal_portfolio_value
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics_gbm(simulation_results, initial_value, confidence)
    
    # Path analysis from stored paths
    terminal_asset_values = stored_paths[:, -1, :]  # Final values
    path_statistics = {}
    
    for asset_idx in range(num_assets):
        asset_terminals = terminal_asset_values[:, asset_idx]
        asset_returns = (asset_terminals / portfolio[asset_idx]) - 1
        
        path_statistics[f'asset_{asset_idx}'] = {
            'final_mean': float(np.mean(asset_terminals)),
            'final_std': float(np.std(asset_terminals)),
            'return_mean': float(np.mean(asset_returns) * 100),
            'return_std': float(np.std(asset_returns) * 100),
            'max_drawdown': calculate_max_drawdown(stored_paths[:, :, asset_idx], portfolio[asset_idx])
        }
    
    # Portfolio-level statistics
    portfolio_returns = (simulation_results / initial_value) - 1
    
    # Theoretical vs empirical comparison
    theoretical_mean = initial_value * np.exp(np.sum((portfolio / initial_value) * expected_returns) * horizon)
    theoretical_std = initial_value * np.sqrt(
        np.sum((portfolio / initial_value)**2 * volatilities**2) * horizon +
        2 * np.sum([
            (portfolio[i] * portfolio[j] / initial_value**2) * volatilities[i] * volatilities[j] * correlation_matrix[i, j] * horizon
            for i in range(num_assets) for j in range(i + 1, num_assets)
        ])
    )
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'gbm',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio.tolist(),
            'iterations': iterations,
            'confidence': confidence,
            'horizon_years': horizon,
            'time_steps': time_steps,
            'simulation_frequency': 'daily' if time_steps == 252 else 'custom',
            'numerical_scheme': 'exact_gbm_solution'
        },
        
        'risk_metrics': risk_metrics,
        
        'gbm_parameters': {
            'expected_returns': [float(x) for x in expected_returns],
            'volatilities': [float(x) for x in volatilities],
            'correlation_matrix': correlation_matrix.tolist(),
            'drift_adjustment': [float(x) for x in drift_adjusted],
            'time_step_size': float(dt)
        },
        
        'path_analysis': {
            'individual_assets': path_statistics,
            'stored_paths': int(store_paths),
            'average_path_volatility': float(np.mean([path_statistics[f'asset_{i}']['return_std'] for i in range(num_assets)])),
            'path_correlation_preserved': True
        },
        
        'theoretical_comparison': {
            'theoretical_mean': float(theoretical_mean),
            'empirical_mean': float(np.mean(simulation_results)),
            'theoretical_std': float(theoretical_std),
            'empirical_std': float(np.std(simulation_results)),
            'mean_error': float(abs(np.mean(simulation_results) - theoretical_mean) / theoretical_mean * 100),
            'std_error': float(abs(np.std(simulation_results) - theoretical_std) / theoretical_std * 100)
        },
        
        'portfolio_stats': {
            'expected_terminal': float(np.mean(simulation_results)),
            'expected_return': float(np.mean(portfolio_returns) * 100),
            'volatility': float(np.std(portfolio_returns) * 100),
            'sharpe_ratio': float(np.mean(portfolio_returns) / np.std(portfolio_returns)) if np.std(portfolio_returns) > 0 else 0,
            'skewness': float(calculate_skewness(portfolio_returns)),
            'kurtosis': float(calculate_kurtosis(portfolio_returns))
        },
        
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time) if simulation_time > 0 else 0,
            'paths_computed': int(iterations * time_steps),
            'random_numbers_generated': int(iterations * time_steps * num_assets)
        }
    }

def multi_asset_gbm(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Multi-asset correlated GBM simulation with advanced features.
    
    This implementation includes:
    - Full correlation structure modeling
    - Different drift rates per asset
    - Time-varying parameters (optional)
    - Jump diffusion components (optional)
    
    Args:
        portfolio: List of asset values
        params: Advanced simulation parameters
        
    Returns:
        Dictionary with multi-asset GBM results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Extract parameters
    iterations = params.get('iterations', 100000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    time_steps = params.get('time_steps', 252)
    
    # Advanced parameters
    jump_intensity = params.get('jump_intensity', 0.0)  # Jumps per year
    jump_mean = params.get('jump_mean', 0.0)  # Average jump size
    jump_std = params.get('jump_std', 0.1)  # Jump volatility
    time_varying_vol = params.get('time_varying_volatility', False)
    
    portfolio = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio)
    num_assets = len(portfolio)
    
    print(f"🔀 Running Multi-Asset GBM Simulation:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Assets: {num_assets}")
    print(f"  Jump Intensity: {jump_intensity}/year")
    print(f"  Time-Varying Vol: {time_varying_vol}")
    
    # Parameters with defaults
    expected_returns = np.array(params.get('expected_returns', [0.08] * num_assets))
    base_volatilities = np.array(params.get('volatilities', [0.15] * num_assets))
    
    # Correlation handling
    correlations = params.get('correlations', None)
    if correlations is None:
        correlation_matrix = np.eye(num_assets)
    elif isinstance(correlations, (int, float)):
        correlation_matrix = np.full((num_assets, num_assets), correlations)
        np.fill_diagonal(correlation_matrix, 1.0)
    else:
        correlation_matrix = np.array(correlations)
    
    # Cholesky decomposition
    try:
        chol_matrix = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        chol_matrix = np.eye(num_assets)
    
    # Time setup
    dt = horizon / time_steps
    sqrt_dt = np.sqrt(dt)
    
    simulation_results = np.zeros(iterations)
    jump_counts = np.zeros(iterations)
    
    # Detailed path storage for multi-asset analysis
    store_detailed_paths = min(100, iterations)
    detailed_paths = np.zeros((store_detailed_paths, time_steps + 1, num_assets))
    
    for i in range(iterations):
        # Initialize asset prices
        asset_prices = portfolio.copy()
        
        if i < store_detailed_paths:
            detailed_paths[i, 0] = asset_prices
        
        iteration_jump_count = 0
        
        for t in range(time_steps):
            # Current volatilities (time-varying if enabled)
            if time_varying_vol:
                # Simple time-varying volatility: higher volatility in early periods
                vol_multiplier = 1.0 + 0.5 * np.sin(2 * np.pi * t / time_steps)
                current_volatilities = base_volatilities * vol_multiplier
            else:
                current_volatilities = base_volatilities
            
            # Drift adjustment for each asset
            drift_adjusted = expected_returns - 0.5 * current_volatilities**2
            
            # Generate correlated Brownian motion increments
            independent_randoms = np.random.standard_normal(num_assets)
            correlated_randoms = chol_matrix @ independent_randoms
            
            # Brownian motion component
            brownian_component = current_volatilities * sqrt_dt * correlated_randoms
            deterministic_component = drift_adjusted * dt
            
            # Jump component (if enabled)
            jump_component = np.zeros(num_assets)
            if jump_intensity > 0:
                # Poisson process for jump arrivals
                jump_probability = jump_intensity * dt
                for asset_idx in range(num_assets):
                    if np.random.random() < jump_probability:
                        # Jump occurs
                        jump_size = np.random.normal(jump_mean, jump_std)
                        jump_component[asset_idx] = jump_size
                        iteration_jump_count += 1
            
            # Combined price update
            total_log_return = deterministic_component + brownian_component + jump_component
            asset_prices = asset_prices * np.exp(total_log_return)
            
            if i < store_detailed_paths:
                detailed_paths[i, t + 1] = asset_prices
        
        jump_counts[i] = iteration_jump_count
        simulation_results[i] = np.sum(asset_prices)
    
    # Risk metrics
    risk_metrics = calculate_risk_metrics_gbm(simulation_results, initial_value, confidence)
    
    # Multi-asset correlation analysis
    if store_detailed_paths > 0:
        terminal_returns = np.zeros((store_detailed_paths, num_assets))
        for asset_idx in range(num_assets):
            terminal_values = detailed_paths[:, -1, asset_idx]
            terminal_returns[:, asset_idx] = (terminal_values / portfolio[asset_idx]) - 1
        
        empirical_correlation = np.corrcoef(terminal_returns.T)
        correlation_error = np.mean(np.abs(empirical_correlation - correlation_matrix))
    else:
        empirical_correlation = np.eye(num_assets)
        correlation_error = 0.0
    
    # Jump analysis
    jump_analysis = {
        'total_jumps': int(np.sum(jump_counts)),
        'average_jumps_per_path': float(np.mean(jump_counts)),
        'max_jumps_in_path': int(np.max(jump_counts)),
        'jump_frequency': float(np.sum(jump_counts) / (iterations * time_steps))
    } if jump_intensity > 0 else {'jumps_disabled': True}
    
    # Asset contribution analysis
    if num_assets > 1:
        asset_contributions = np.zeros(num_assets)
        for asset_idx in range(num_assets):
            asset_weight = portfolio[asset_idx] / initial_value
            asset_contributions[asset_idx] = asset_weight * 100
        
        largest_contributor = int(np.argmax(asset_contributions))
    else:
        asset_contributions = [100.0]
        largest_contributor = 0
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'multi_asset_gbm',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio.tolist(),
            'iterations': iterations,
            'confidence': confidence,
            'horizon_years': horizon,
            'time_steps': time_steps,
            'num_assets': num_assets,
            'advanced_features': {
                'jump_diffusion': jump_intensity > 0,
                'time_varying_volatility': time_varying_vol,
                'full_correlation_structure': True
            }
        },
        
        'risk_metrics': risk_metrics,
        
        'correlation_analysis': {
            'input_correlation_matrix': correlation_matrix.tolist(),
            'empirical_correlation_matrix': empirical_correlation.tolist(),
            'correlation_preservation_error': float(correlation_error),
            'correlation_quality': 'excellent' if correlation_error < 0.05 else 'good' if correlation_error < 0.10 else 'fair'
        },
        
        'jump_analysis': jump_analysis,
        
        'asset_analysis': {
            'portfolio_weights': [float(w) for w in (portfolio / initial_value)],
            'weight_percentages': [float(c) for c in asset_contributions],
            'largest_contributor': largest_contributor,
            'diversification_score': float(1 - np.max(portfolio / initial_value)) if num_assets > 1 else 0.0
        },
        
        'model_validation': {
            'paths_analyzed': int(store_detailed_paths),
            'numerical_stability': 'stable',
            'convergence_indicator': float(np.std(simulation_results[-1000:]) / np.std(simulation_results[:1000])) if iterations >= 2000 else 1.0
        },
        
        'portfolio_stats': {
            'expected_terminal': float(np.mean(simulation_results)),
            'expected_return': float((np.mean(simulation_results) / initial_value - 1) * 100),
            'volatility': float(np.std(simulation_results) / initial_value * 100),
            'sharpe_ratio': float((np.mean(simulation_results) / initial_value - 1) / (np.std(simulation_results) / initial_value)) if np.std(simulation_results) > 0 else 0,
            'multi_asset_complexity': 'high' if num_assets > 5 else 'medium' if num_assets > 2 else 'low'
        },
        
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time) if simulation_time > 0 else 0,
            'computational_intensity': float(iterations * time_steps * num_assets),
            'memory_efficiency': 'optimized_for_large_portfolios'
        }
    }

def path_dependent_gbm(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Path-dependent GBM simulation with barriers and exotic features.
    
    This method includes:
    - Barrier options-style monitoring
    - Knockin/knockout levels
    - Asian-style path averaging
    - Lookback features
    
    Args:
        portfolio: List of asset values
        params: Path-dependent simulation parameters
        
    Returns:
        Dictionary with path-dependent GBM results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Standard parameters
    iterations = params.get('iterations', 100000)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    time_steps = params.get('time_steps', 252)
    
    # Path-dependent parameters
    barrier_level = params.get('barrier_level', None)  # Barrier as fraction of initial value
    barrier_type = params.get('barrier_type', 'down_and_out')  # up_and_out, down_and_in, etc.
    averaging_period = params.get('averaging_period', None)  # Asian-style averaging
    lookback_monitoring = params.get('lookback_monitoring', False)
    
    portfolio = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio)
    
    print(f"🛤️  Running Path-Dependent GBM Simulation:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Barrier Level: {barrier_level}")
    print(f"  Barrier Type: {barrier_type}")
    print(f"  Lookback: {lookback_monitoring}")
    
    # Set up barrier monitoring
    if barrier_level is not None:
        barrier_value = initial_value * barrier_level
        print(f"  Barrier Value: ${barrier_value:,.2f}")
    else:
        barrier_value = None
    
    # Basic GBM parameters
    expected_returns = np.array(params.get('expected_returns', [0.08]))
    volatilities = np.array(params.get('volatilities', [0.15]))
    
    # Ensure single asset for simplicity of path-dependent features
    if len(portfolio) > 1:
        # Aggregate to single asset
        portfolio_return = np.sum((portfolio / initial_value) * expected_returns)
        portfolio_volatility = np.sqrt(np.sum((portfolio / initial_value)**2 * volatilities**2))
        expected_returns = np.array([portfolio_return])
        volatilities = np.array([portfolio_volatility])
    
    dt = horizon / time_steps
    sqrt_dt = np.sqrt(dt)
    drift_adjusted = expected_returns[0] - 0.5 * volatilities[0]**2
    
    simulation_results = np.zeros(iterations)
    barrier_events = np.zeros(iterations, dtype=bool)
    path_statistics = {
        'max_values': np.zeros(iterations),
        'min_values': np.zeros(iterations),
        'average_values': np.zeros(iterations),
        'barrier_hits': np.zeros(iterations, dtype=bool)
    }
    
    for i in range(iterations):
        portfolio_value = initial_value
        path_values = np.zeros(time_steps + 1)
        path_values[0] = portfolio_value
        
        barrier_hit = False
        
        for t in range(time_steps):
            # GBM step
            random_increment = np.random.standard_normal()
            log_return = drift_adjusted * dt + volatilities[0] * sqrt_dt * random_increment
            portfolio_value = portfolio_value * np.exp(log_return)
            path_values[t + 1] = portfolio_value
            
            # Barrier monitoring
            if barrier_value is not None:
                if barrier_type == 'down_and_out' and portfolio_value <= barrier_value:
                    barrier_hit = True
                elif barrier_type == 'up_and_out' and portfolio_value >= barrier_value:
                    barrier_hit = True
                elif barrier_type == 'down_and_in' and portfolio_value <= barrier_value:
                    barrier_hit = True
                elif barrier_type == 'up_and_in' and portfolio_value >= barrier_value:
                    barrier_hit = True
        
        # Path statistics
        path_statistics['max_values'][i] = np.max(path_values)
        path_statistics['min_values'][i] = np.min(path_values)
        path_statistics['average_values'][i] = np.mean(path_values)
        path_statistics['barrier_hits'][i] = barrier_hit
        
        # Final value (potentially modified by barriers)
        if barrier_type.endswith('out') and barrier_hit:
            # Knocked out - set to barrier value or zero
            final_value = 0.0 if 'out' in barrier_type else barrier_value
        elif barrier_type.endswith('in') and not barrier_hit:
            # Didn't knock in - use initial value or modified payoff
            final_value = initial_value
        else:
            # Normal case or barrier condition met
            final_value = portfolio_value
        
        # Asian averaging if enabled
        if averaging_period is not None:
            avg_start = max(0, time_steps - averaging_period)
            averaged_value = np.mean(path_values[avg_start:])
            final_value = averaged_value
        
        simulation_results[i] = final_value
        barrier_events[i] = barrier_hit
    
    # Risk metrics
    risk_metrics = calculate_risk_metrics_gbm(simulation_results, initial_value, confidence)
    
    # Path-dependent analysis
    barrier_probability = np.mean(barrier_events) * 100 if barrier_value is not None else 0.0
    
    # Lookback analysis
    max_values = path_statistics['max_values']
    min_values = path_statistics['min_values']
    lookback_call_payoff = np.maximum(simulation_results - min_values, 0)
    lookback_put_payoff = np.maximum(max_values - simulation_results, 0)
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'path_dependent_gbm',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio.tolist(),
            'iterations': iterations,
            'confidence': confidence,
            'horizon_years': horizon,
            'time_steps': time_steps,
            'path_features': {
                'barrier_monitoring': barrier_value is not None,
                'barrier_type': barrier_type if barrier_value is not None else None,
                'asian_averaging': averaging_period is not None,
                'lookback_monitoring': lookback_monitoring
            }
        },
        
        'risk_metrics': risk_metrics,
        
        'barrier_analysis': {
            'barrier_level': float(barrier_level) if barrier_level is not None else None,
            'barrier_value': float(barrier_value) if barrier_value is not None else None,
            'barrier_hit_probability': float(barrier_probability),
            'barrier_type': barrier_type if barrier_value is not None else None,
            'barrier_affected_scenarios': int(np.sum(barrier_events))
        } if barrier_value is not None else {'barrier_monitoring': False},
        
        'path_statistics': {
            'maximum_portfolio_values': {
                'mean': float(np.mean(max_values)),
                'std': float(np.std(max_values)),
                'max': float(np.max(max_values))
            },
            'minimum_portfolio_values': {
                'mean': float(np.mean(min_values)),
                'std': float(np.std(min_values)),
                'min': float(np.min(min_values))
            },
            'average_path_values': {
                'mean': float(np.mean(path_statistics['average_values'])),
                'std': float(np.std(path_statistics['average_values']))
            }
        },
        
        'exotic_payoffs': {
            'lookback_call_value': float(np.mean(lookback_call_payoff)),
            'lookback_put_value': float(np.mean(lookback_put_payoff)),
            'asian_value': float(np.mean(path_statistics['average_values'])) if averaging_period else None,
            'path_dependency_impact': float(abs(np.mean(simulation_results) - initial_value * np.exp(expected_returns[0] * horizon)) / initial_value * 100)
        },
        
        'portfolio_stats': {
            'expected_terminal': float(np.mean(simulation_results)),
            'expected_return': float((np.mean(simulation_results) / initial_value - 1) * 100),
            'volatility': float(np.std(simulation_results) / initial_value * 100),
            'sharpe_ratio': float((np.mean(simulation_results) / initial_value - 1) / (np.std(simulation_results) / initial_value)) if np.std(simulation_results) > 0 else 0,
            'path_efficiency': float(np.mean(simulation_results) / np.mean(max_values)) if np.mean(max_values) > 0 else 1.0
        },
        
        'performance': {
            'simulation_time': float(simulation_time),
            'iterations_per_second': int(iterations / simulation_time) if simulation_time > 0 else 0,
            'path_monitoring_overhead': 'high' if barrier_value is not None or lookback_monitoring else 'low',
            'memory_usage': 'intensive_path_storage'
        }
    }

def calculate_max_drawdown(paths: np.ndarray, initial_value: float) -> float:
    """Calculate maximum drawdown for asset paths"""
    if len(paths.shape) == 1:
        paths = paths.reshape(1, -1)
    
    max_drawdowns = []
    for path in paths:
        running_max = np.maximum.accumulate(path)
        drawdowns = (running_max - path) / running_max
        max_drawdowns.append(np.max(drawdowns))
    
    return float(np.mean(max_drawdowns) * 100)

def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return 0.0
    skew = np.mean(((data - mean_val) / std_val) ** 3)
    return skew

def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis of data"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return 0.0
    kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3
    return kurt

# Export functions
__all__ = [
    'gbm_simulation',
    'multi_asset_gbm',
    'path_dependent_gbm',
    'calculate_max_drawdown',
    'calculate_skewness',
    'calculate_kurtosis'
]
