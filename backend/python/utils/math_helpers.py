"""
Mathematical Helper Functions for Financial Risk Simulations - Python Implementation

This module provides core mathematical utilities used across different
simulation strategies including random number generation, statistical
functions, and financial calculations.
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Optional
import warnings

def random_normal(mean: float = 0, std: float = 1, size: Optional[int] = None) -> np.ndarray:
    """
    Generate random number(s) from normal distribution
    
    Args:
        mean: Mean of the distribution (default: 0)
        std: Standard deviation (default: 1)
        size: Number of samples (None for single value)
    
    Returns:
        Random number(s) from normal distribution
    """
    return np.random.normal(mean, std, size)

def calculate_percentile(arr: np.ndarray, percentile: float) -> float:
    """
    Calculate percentile of an array
    
    Args:
        arr: Array of numbers
        percentile: Percentile to calculate (0-100)
    
    Returns:
        Value at the specified percentile
    """
    return np.percentile(arr, percentile)

def calculate_var(losses: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) from a distribution of losses
    
    Args:
        losses: Array of loss values
        confidence: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        VaR value
    """
    percentile_value = confidence * 100
    return calculate_percentile(losses, percentile_value)

def calculate_cvar(losses: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) - Expected Shortfall
    
    Args:
        losses: Array of loss values
        confidence: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        CVaR value
    """
    var_threshold = calculate_var(losses, confidence)
    tail_losses = losses[losses >= var_threshold]
    
    if len(tail_losses) == 0:
        return var_threshold
    
    return np.mean(tail_losses)

def calculate_portfolio_stats(weights: np.ndarray, 
                            expected_returns: np.ndarray, 
                            covariance_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate portfolio statistics (mean, variance, volatility)
    
    Args:
        weights: Portfolio weights (must sum to 1)
        expected_returns: Expected returns for each asset
        covariance_matrix: Covariance matrix
    
    Returns:
        Dictionary with portfolio statistics
    """
    # Portfolio expected return: w^T * μ
    portfolio_return = np.dot(weights, expected_returns)
    
    # Portfolio variance: w^T * Σ * w
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    return {
        'expected_return': float(portfolio_return),
        'variance': float(portfolio_variance),
        'volatility': float(portfolio_volatility)
    }

def correlation_to_covariance(volatilities: np.ndarray, 
                            correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Create covariance matrix from volatilities and correlation coefficients
    
    Args:
        volatilities: Array of asset volatilities
        correlation_matrix: Correlation matrix
    
    Returns:
        Covariance matrix
    """
    vol_matrix = np.diag(volatilities)
    return np.dot(vol_matrix, np.dot(correlation_matrix, vol_matrix))

def cholesky_decomposition(matrix: np.ndarray) -> np.ndarray:
    """
    Perform Cholesky decomposition for correlated random variables
    
    Args:
        matrix: Positive definite matrix
    
    Returns:
        Lower triangular matrix L where A = L * L^T
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # If matrix is not positive definite, try to fix it
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
        fixed_matrix = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
        return np.linalg.cholesky(fixed_matrix)

def generate_correlated_normals(size: int, covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate correlated random variables using Cholesky decomposition
    
    Args:
        size: Number of samples to generate
        covariance_matrix: Covariance matrix
    
    Returns:
        Array of correlated random vectors
    """
    n = covariance_matrix.shape[0]
    
    # Generate independent standard normals
    independent = np.random.standard_normal((size, n))
    
    # Cholesky decomposition
    L = cholesky_decomposition(covariance_matrix)
    
    # Transform to correlated variables: Y = X * L^T
    correlated = np.dot(independent, L.T)
    
    return correlated

def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistical measures
    
    Args:
        data: Array of numerical data
    
    Returns:
        Dictionary with statistical measures
    """
    if len(data) == 0:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 
            'median': 0.0, 'variance': 0.0, 'count': 0
        }
    
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'variance': float(np.var(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'count': len(data)
    }

def bootstrap_sample(data: np.ndarray, sample_size: Optional[int] = None) -> np.ndarray:
    """
    Bootstrap sampling - sample with replacement
    
    Args:
        data: Original data array
        sample_size: Size of bootstrap sample (default: same as original)
    
    Returns:
        Bootstrap sample
    """
    if sample_size is None:
        sample_size = len(data)
    
    indices = np.random.choice(len(data), size=sample_size, replace=True)
    return data[indices]

def normal_cdf(z: float) -> float:
    """
    Standard normal cumulative distribution function
    
    Args:
        z: Z-score
    
    Returns:
        Cumulative probability
    """
    return stats.norm.cdf(z)

def normal_inverse_cdf(p: float) -> float:
    """
    Inverse normal cumulative distribution function
    
    Args:
        p: Probability (0 < p < 1)
    
    Returns:
        Z-score
    """
    return stats.norm.ppf(p)

def calculate_skewness(data: np.ndarray) -> float:
    """
    Calculate skewness of a data series
    
    Args:
        data: Data array
    
    Returns:
        Skewness value
    """
    return float(stats.skew(data))

def calculate_kurtosis(data: np.ndarray) -> float:
    """
    Calculate kurtosis of a data series
    
    Args:
        data: Data array
    
    Returns:
        Kurtosis value
    """
    return float(stats.kurtosis(data))

def calculate_autocorrelation(data: np.ndarray, max_lag: int = 5) -> List[float]:
    """
    Calculate autocorrelation for different lags
    
    Args:
        data: Time series data
        max_lag: Maximum lag to calculate
    
    Returns:
        List of autocorrelation values
    """
    n = len(data)
    data_centered = data - np.mean(data)
    autocorrelations = []
    
    for lag in range(1, max_lag + 1):
        if lag >= n:
            autocorrelations.append(0.0)
            continue
            
        numerator = np.sum(data_centered[:-lag] * data_centered[lag:])
        denominator = np.sum(data_centered ** 2)
        
        correlation = numerator / denominator if denominator > 0 else 0.0
        autocorrelations.append(float(correlation))
    
    return autocorrelations

def calculate_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Calculate correlation between two return series
    
    Args:
        series1: First return series
        series2: Second return series
    
    Returns:
        Correlation coefficient
    """
    return float(np.corrcoef(series1, series2)[0, 1])

def validate_portfolio(portfolio: List[float]) -> np.ndarray:
    """
    Validate and convert portfolio to numpy array
    
    Args:
        portfolio: List of portfolio values
    
    Returns:
        Validated numpy array
    
    Raises:
        ValueError: If portfolio is invalid
    """
    portfolio_array = np.array(portfolio)
    
    if len(portfolio_array) == 0:
        raise ValueError("Portfolio cannot be empty")
    
    if np.any(portfolio_array <= 0):
        raise ValueError("All portfolio values must be positive")
    
    if not np.all(np.isfinite(portfolio_array)):
        raise ValueError("Portfolio values must be finite numbers")
    
    return portfolio_array

def validate_correlation_matrix(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Validate correlation matrix
    
    Args:
        corr_matrix: Correlation matrix to validate
    
    Returns:
        Validated correlation matrix
    
    Raises:
        ValueError: If matrix is invalid
    """
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    
    if not np.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("Correlation matrix must be symmetric")
    
    if not np.allclose(np.diag(corr_matrix), 1.0):
        raise ValueError("Correlation matrix diagonal must be 1.0")
    
    if np.any(np.abs(corr_matrix) > 1.0):
        raise ValueError("Correlation values must be between -1 and 1")
    
    # Check if positive semi-definite
    eigenvals = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvals < -1e-8):
        warnings.warn("Correlation matrix is not positive semi-definite")
    
    return corr_matrix

def ensure_positive_definite(matrix: np.ndarray, min_eigenval: float = 1e-8) -> np.ndarray:
    """
    Ensure matrix is positive definite by adjusting eigenvalues
    
    Args:
        matrix: Input matrix
        min_eigenval: Minimum eigenvalue to enforce
    
    Returns:
        Positive definite matrix
    """
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    eigenvals = np.maximum(eigenvals, min_eigenval)
    return np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
