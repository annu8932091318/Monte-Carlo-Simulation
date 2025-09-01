"""
Variance-Covariance (VarCov) Simulation Module

This module implements the variance-covariance method for risk assessment.
The VarCov method uses analytical formulas based on the assumption of
multivariate normal returns to calculate risk metrics. It's computationally
efficient but relies on normality assumptions.

Key Features:
- Analytical VaR and CVaR calculation
- Multivariate normal distribution modeling
- Stress testing capabilities
- Portfolio optimization integration
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings
try:
    from scipy import stats
    from scipy.linalg import sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available, using NumPy alternatives")

def calculate_portfolio_statistics_local(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    horizon: float = 1.0
) -> Dict[str, float]:
    """Calculate portfolio statistics using matrix operations"""
    
    # Portfolio expected return
    portfolio_return = np.sum(weights * expected_returns) * horizon
    
    # Portfolio variance
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights)) * horizon
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    return {
        'expected_return': float(portfolio_return),
        'volatility': float(portfolio_volatility),
        'variance': float(portfolio_variance)
    }

def variance_covariance_method(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Variance-Covariance method for portfolio risk assessment.
    
    This method calculates risk metrics analytically using the assumption
    of multivariate normal returns. It's fast and provides exact results
    under normality but may underestimate tail risks.
    
    Args:
        portfolio: List of asset values
        params: Parameters including expected returns, volatilities, correlations
        
    Returns:
        Dictionary with VarCov risk assessment results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    expected_returns = params.get('expected_returns', None)
    volatilities = params.get('volatilities', None)
    correlations = params.get('correlations', None)
    
    portfolio_array = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio_array)
    num_assets = len(portfolio_array)
    weights = portfolio_array / initial_value
    
    print(f"📊 Running Variance-Covariance Method:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Confidence Level: {confidence:.1%}")
    print(f"  Time Horizon: {horizon} years")
    print(f"  Assets: {num_assets}")
    
    # Set default parameters if not provided
    if expected_returns is None:
        expected_returns = np.array([0.08] * num_assets)  # 8% annual return
    else:
        expected_returns = np.array(expected_returns)
    
    if volatilities is None:
        volatilities = np.array([0.15] * num_assets)  # 15% annual volatility
    else:
        volatilities = np.array(volatilities)
    
    # Create correlation matrix
    if correlations is None:
        correlation_matrix = np.eye(num_assets)  # No correlation
    elif isinstance(correlations, (int, float)):
        correlation_matrix = np.full((num_assets, num_assets), correlations)
        np.fill_diagonal(correlation_matrix, 1.0)
    else:
        correlation_matrix = np.array(correlations)
    
    # Ensure correlation matrix is positive definite
    eigenvals = np.linalg.eigvals(correlation_matrix)
    if np.any(eigenvals <= 0):
        print("  ⚠️  Correlation matrix not positive definite, regularizing...")
        correlation_matrix = correlation_matrix + np.eye(num_assets) * 1e-6
    
    # Create covariance matrix
    volatility_matrix = np.diag(volatilities)
    covariance_matrix = volatility_matrix @ correlation_matrix @ volatility_matrix
    
    # Calculate portfolio statistics
    portfolio_stats = calculate_portfolio_statistics_local(
        weights, expected_returns, covariance_matrix, horizon
    )
    
    portfolio_return = portfolio_stats['expected_return']
    portfolio_volatility = portfolio_stats['volatility']
    
    # Calculate VaR and CVaR analytically
    if SCIPY_AVAILABLE:
        # Use scipy for more accurate normal distribution calculations
        alpha = 1 - confidence
        z_alpha = stats.norm.ppf(alpha)
        
        # VaR calculation (negative of the percentile)
        var_dollar = initial_value * (-portfolio_return + portfolio_volatility * (-z_alpha))
        
        # CVaR calculation (expected shortfall)
        phi_z = stats.norm.pdf(z_alpha)
        cvar_multiplier = phi_z / alpha
        cvar_dollar = initial_value * (-portfolio_return + portfolio_volatility * cvar_multiplier)
        
    else:
        # Use NumPy approximation
        alpha = 1 - confidence
        
        # Approximate normal quantile (works well for common confidence levels)
        if confidence == 0.95:
            z_alpha = -1.645
        elif confidence == 0.99:
            z_alpha = -2.326
        else:
            # Rough approximation for other confidence levels
            z_alpha = -np.sqrt(2) * (1 - 2 * alpha)
        
        var_dollar = initial_value * (-portfolio_return + portfolio_volatility * (-z_alpha))
        
        # Approximate CVaR
        phi_z = np.exp(-0.5 * z_alpha**2) / np.sqrt(2 * np.pi)
        cvar_multiplier = phi_z / alpha
        cvar_dollar = initial_value * (-portfolio_return + portfolio_volatility * cvar_multiplier)
    
    # Additional analytical calculations
    expected_terminal_value = initial_value * (1 + portfolio_return)
    
    # Portfolio diversification analysis
    individual_vars = []
    for i in range(num_assets):
        asset_weight = weights[i]
        asset_return = expected_returns[i] * horizon
        asset_volatility = volatilities[i] * np.sqrt(horizon)
        
        if SCIPY_AVAILABLE:
            asset_var = (portfolio_array[i]) * (-asset_return + asset_volatility * (-stats.norm.ppf(alpha)))
        else:
            asset_var = (portfolio_array[i]) * (-asset_return + asset_volatility * (-z_alpha))
        
        individual_vars.append(asset_var)
    
    undiversified_var = np.sum(individual_vars)
    diversification_benefit = (undiversified_var - var_dollar) / undiversified_var * 100 if undiversified_var > 0 else 0
    
    # Risk decomposition
    marginal_vars = np.zeros(num_assets)
    component_vars = np.zeros(num_assets)
    
    for i in range(num_assets):
        # Marginal VaR: change in portfolio VaR for small change in asset weight
        portfolio_cov_i = np.sum(covariance_matrix[i] * weights)
        marginal_vars[i] = (-z_alpha if not SCIPY_AVAILABLE else -stats.norm.ppf(alpha)) * portfolio_cov_i * np.sqrt(horizon) * initial_value
        
        # Component VaR: contribution of asset i to total VaR
        component_vars[i] = weights[i] * marginal_vars[i]
    
    simulation_time = time.time() - start_time
    
    return {
        'simulation_config': {
            'method': 'variance_covariance',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio_array.tolist(),
            'confidence': confidence,
            'horizon_years': horizon,
            'assumption': 'multivariate_normal_returns',
            'computation_type': 'analytical'
        },
        
        'risk_metrics': {
            'VaR_95': float(var_dollar),
            'CVaR_95': float(cvar_dollar),
            'probability_of_loss': float(50.0),  # Under normal assumption, approximately 50%
            'expected_shortfall': float(cvar_dollar),
            'maximum_loss': float(float('inf')),  # Theoretically unbounded under normal assumption
            'maximum_gain': float(float('inf'))
        },
        
        'portfolio_analytics': {
            'expected_terminal_value': float(expected_terminal_value),
            'portfolio_return': float(portfolio_return * 100),
            'portfolio_volatility': float(portfolio_volatility * 100),
            'sharpe_ratio': float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0,
            'diversification_benefit': float(diversification_benefit),
            'undiversified_VaR': float(undiversified_var)
        },
        
        'risk_decomposition': {
            'marginal_VaRs': [float(x) for x in marginal_vars],
            'component_VaRs': [float(x) for x in component_vars],
            'component_percentages': [float(x / var_dollar * 100) if var_dollar > 0 else 0 for x in component_vars],
            'largest_contributor': int(np.argmax(np.abs(component_vars))) if len(component_vars) > 0 else 0
        },
        
        'model_parameters': {
            'expected_returns': [float(x) for x in expected_returns],
            'volatilities': [float(x) for x in volatilities],
            'correlation_matrix': correlation_matrix.tolist(),
            'covariance_matrix': covariance_matrix.tolist(),
            'weights': [float(x) for x in weights]
        },
        
        'performance': {
            'calculation_time': float(simulation_time),
            'computational_complexity': 'O(n²)',
            'matrix_operations': num_assets ** 2,
            'analytical_precision': 'exact_under_normality'
        }
    }

def stress_test_varcov(
    portfolio: List[float],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Stress testing using the Variance-Covariance method.
    
    This method performs various stress tests on the portfolio by
    modifying key parameters and recalculating risk metrics to
    understand portfolio sensitivity.
    
    Args:
        portfolio: List of asset values
        params: Parameters including stress test scenarios
        
    Returns:
        Dictionary with stress test results
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    portfolio_array = np.array(portfolio, dtype=np.float64)
    initial_value = np.sum(portfolio_array)
    num_assets = len(portfolio_array)
    
    print(f"🧪 Running VarCov Stress Testing:")
    print(f"  Portfolio Value: ${initial_value:,.2f}")
    print(f"  Stress Scenarios: Multiple")
    
    # Base case parameters
    base_expected_returns = params.get('expected_returns', [0.08] * num_assets)
    base_volatilities = params.get('volatilities', [0.15] * num_assets)
    base_correlations = params.get('correlations', 0.3)
    confidence = params.get('confidence', 0.95)
    horizon = params.get('horizon', 1.0)
    
    # Calculate base case
    base_result = variance_covariance_method(portfolio, params)
    base_var = base_result['risk_metrics']['VaR_95']
    base_cvar = base_result['risk_metrics']['CVaR_95']
    
    stress_results = {}
    
    # Stress Test 1: Volatility shock (+50% volatility)
    print("  Running volatility shock test...")
    stress_vol_params = params.copy()
    stress_vol_params['volatilities'] = [v * 1.5 for v in base_volatilities]
    vol_stress_result = variance_covariance_method(portfolio, stress_vol_params)
    
    stress_results['volatility_shock'] = {
        'scenario': 'Volatility +50%',
        'VaR_95': vol_stress_result['risk_metrics']['VaR_95'],
        'CVaR_95': vol_stress_result['risk_metrics']['CVaR_95'],
        'VaR_change': float((vol_stress_result['risk_metrics']['VaR_95'] - base_var) / base_var * 100),
        'CVaR_change': float((vol_stress_result['risk_metrics']['CVaR_95'] - base_cvar) / base_cvar * 100)
    }
    
    # Stress Test 2: Return shock (-5% annual returns)
    print("  Running return shock test...")
    stress_ret_params = params.copy()
    stress_ret_params['expected_returns'] = [r - 0.05 for r in base_expected_returns]
    ret_stress_result = variance_covariance_method(portfolio, stress_ret_params)
    
    stress_results['return_shock'] = {
        'scenario': 'Returns -5%',
        'VaR_95': ret_stress_result['risk_metrics']['VaR_95'],
        'CVaR_95': ret_stress_result['risk_metrics']['CVaR_95'],
        'VaR_change': float((ret_stress_result['risk_metrics']['VaR_95'] - base_var) / base_var * 100),
        'CVaR_change': float((ret_stress_result['risk_metrics']['CVaR_95'] - base_cvar) / base_cvar * 100)
    }
    
    # Stress Test 3: Correlation shock (perfect correlation)
    print("  Running correlation shock test...")
    stress_corr_params = params.copy()
    stress_corr_params['correlations'] = 0.9  # High correlation
    corr_stress_result = variance_covariance_method(portfolio, stress_corr_params)
    
    stress_results['correlation_shock'] = {
        'scenario': 'Correlation → 0.9',
        'VaR_95': corr_stress_result['risk_metrics']['VaR_95'],
        'CVaR_95': corr_stress_result['risk_metrics']['CVaR_95'],
        'VaR_change': float((corr_stress_result['risk_metrics']['VaR_95'] - base_var) / base_var * 100),
        'CVaR_change': float((corr_stress_result['risk_metrics']['CVaR_95'] - base_cvar) / base_cvar * 100)
    }
    
    # Stress Test 4: Combined worst case
    print("  Running combined stress test...")
    combined_params = params.copy()
    combined_params['volatilities'] = [v * 1.5 for v in base_volatilities]
    combined_params['expected_returns'] = [r - 0.05 for r in base_expected_returns]
    combined_params['correlations'] = 0.9
    combined_result = variance_covariance_method(portfolio, combined_params)
    
    stress_results['combined_worst'] = {
        'scenario': 'Vol +50%, Returns -5%, Corr 0.9',
        'VaR_95': combined_result['risk_metrics']['VaR_95'],
        'CVaR_95': combined_result['risk_metrics']['CVaR_95'],
        'VaR_change': float((combined_result['risk_metrics']['VaR_95'] - base_var) / base_var * 100),
        'CVaR_change': float((combined_result['risk_metrics']['CVaR_95'] - base_cvar) / base_cvar * 100)
    }
    
    # Stress Test 5: Time horizon sensitivity
    horizon_tests = {}
    for test_horizon in [0.25, 0.5, 2.0, 5.0]:  # 3 months, 6 months, 2 years, 5 years
        horizon_params = params.copy()
        horizon_params['horizon'] = test_horizon
        horizon_result = variance_covariance_method(portfolio, horizon_params)
        
        horizon_tests[f'horizon_{test_horizon}y'] = {
            'VaR_95': horizon_result['risk_metrics']['VaR_95'],
            'CVaR_95': horizon_result['risk_metrics']['CVaR_95']
        }
    
    # Sensitivity analysis summary
    var_sensitivities = {
        'volatility_sensitivity': stress_results['volatility_shock']['VaR_change'],
        'return_sensitivity': stress_results['return_shock']['VaR_change'],
        'correlation_sensitivity': stress_results['correlation_shock']['VaR_change'],
        'combined_sensitivity': stress_results['combined_worst']['VaR_change']
    }
    
    most_sensitive_factor = max(var_sensitivities.items(), key=lambda x: abs(x[1]))
    
    simulation_time = time.time() - start_time
    
    return {
        'stress_test_config': {
            'method': 'stress_test_varcov',
            'engine': 'python',
            'initial_value': float(initial_value),
            'portfolio': portfolio_array.tolist(),
            'base_case_VaR': float(base_var),
            'base_case_CVaR': float(base_cvar),
            'confidence': confidence,
            'horizon_years': horizon
        },
        
        'stress_scenarios': stress_results,
        
        'horizon_sensitivity': horizon_tests,
        
        'sensitivity_analysis': {
            'factor_sensitivities': var_sensitivities,
            'most_sensitive_factor': most_sensitive_factor[0],
            'maximum_sensitivity': float(most_sensitive_factor[1]),
            'resilience_score': float(max(0, 100 - abs(most_sensitive_factor[1])))
        },
        
        'risk_summary': {
            'base_case': {
                'VaR_95': float(base_var),
                'CVaR_95': float(base_cvar)
            },
            'worst_case': {
                'VaR_95': float(combined_result['risk_metrics']['VaR_95']),
                'CVaR_95': float(combined_result['risk_metrics']['CVaR_95'])
            },
            'stress_amplification': float(combined_result['risk_metrics']['VaR_95'] / base_var) if base_var > 0 else 1.0
        },
        
        'performance': {
            'stress_test_time': float(simulation_time),
            'scenarios_tested': len(stress_results) + len(horizon_tests),
            'analytical_efficiency': 'high',
            'computation_type': 'closed_form_solutions'
        }
    }

# Export functions
__all__ = [
    'variance_covariance_method',
    'stress_test_varcov',
    'calculate_portfolio_statistics_local'
]
