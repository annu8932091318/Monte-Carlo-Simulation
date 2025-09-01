"""
Monte Carlo Portfolio Risk Simulation using Geometric Brownian Motion (GBM)

This project implements a comprehensive Monte Carlo simulation framework for portfolio
risk assessment. The simulation models portfolio returns using Geometric Brownian Motion,
which is widely used in quantitative finance for modeling asset prices.

Key Features:
- GPU acceleration support with CuPy (falls back to NumPy if GPU unavailable)
- Configurable portfolio weights, expected returns, volatilities, and correlations
- Risk metrics calculation including VaR, CVaR, and loss probabilities
- High-performance simulation capable of handling millions of scenarios
- Environment variable configuration for easy parameter adjustment

Mathematical Foundation:
The simulation uses the GBM model where portfolio value S(t) follows:
dS(t) = μ * S(t) * dt + σ * S(t) * dW(t)

Where:
- μ (mu) is the drift parameter (expected return)
- σ (sigma) is the volatility parameter
- dW(t) is a Wiener process (Brownian motion)

The analytical solution for terminal value S(T) given initial value S(0) is:
S(T) = S(0) * exp((μ - σ²/2) * T + σ * √T * Z)

Where Z ~ N(0,1) is a standard normal random variable.
"""

import os, time, math
import numpy as _np

# GPU Acceleration Setup
# Try to import CuPy for GPU acceleration, fall back to NumPy if unavailable
try:
    import cupy as cp
    xp = cp  # Use CuPy as the array library
    on_gpu = True
    print("GPU acceleration enabled with CuPy")
except ImportError:
    xp = _np  # Use NumPy as the array library
    on_gpu = False
    print("Using CPU with NumPy (CuPy not available)")

def percentile(x, q):
    """
    Calculate percentile using appropriate library (CuPy or NumPy)
    
    Args:
        x: Array of values
        q: Percentile to calculate (0-100)
    
    Returns:
        Percentile value
    """
    return cp.percentile(x, q) if on_gpu else _np.percentile(x, q)

def to_cpu(x):
    """
    Convert array to CPU memory if using GPU, otherwise return as-is
    
    Args:
        x: Array (CuPy or NumPy)
    
    Returns:
        NumPy array on CPU
    """
    return cp.asnumpy(x) if on_gpu else x

def main(
    n_paths: int = 10_000_000,
    initial_value: float = 100_000.0,
    horizon_years: float = 1.0,
    seed: int = 42
):
    """
    Main Monte Carlo simulation function for portfolio risk assessment.
    
    This function simulates portfolio terminal values using Geometric Brownian Motion (GBM)
    and calculates comprehensive risk metrics including Value at Risk (VaR), Conditional
    Value at Risk (CVaR), and loss probabilities.
    
    The simulation assumes that portfolio returns follow a multivariate normal distribution
    and that individual asset prices follow geometric Brownian motion. The correlation
    structure between assets is modeled through a correlation matrix.
    
    Args:
        n_paths (int): Number of Monte Carlo simulation paths (default: 10,000,000)
                      More paths = higher accuracy but longer computation time
        initial_value (float): Initial portfolio value in dollars (default: $100,000)
        horizon_years (float): Investment horizon in years (default: 1.0 year)
        seed (int): Random seed for reproducible results (default: 42)
    
    Portfolio Construction:
        The portfolio consists of 4 assets with the following characteristics:
        - Asset weights must sum to 1.0 (100% allocation)
        - Expected annual returns (mu) represent the drift component
        - Annual volatilities (sigma) represent the diffusion component
        - Correlation matrix captures the interdependence between assets
    
    Mathematical Model:
        Portfolio return ~ N(μ_p, σ_p²) where:
        - μ_p = w^T * μ (weighted average of expected returns)
        - σ_p² = w^T * Σ * w (portfolio variance from correlation matrix)
        - Σ = D * ρ * D (covariance matrix from correlation and volatilities)
    
    Risk Metrics Calculated:
        - VaR(95%): Value at Risk - potential loss exceeded only 5% of the time
        - CVaR(95%): Conditional VaR - average loss in the worst 5% of scenarios
        - P(Loss): Probability of any loss occurring
        - Percentiles: 10th, 50th (median), and 90th percentiles of terminal values
    """

    # =============================================================================
    # PORTFOLIO CONFIGURATION
    # =============================================================================
    
    """
    Portfolio Asset Allocation:
    Define the portfolio composition with 4 different asset classes.
    In practice, these could represent:
    - Asset 1 (30%): Large-cap equity (moderate return, moderate risk)
    - Asset 2 (25%): Government bonds (low return, low risk)  
    - Asset 3 (25%): Small-cap equity (high return, high risk)
    - Asset 4 (20%): Cash/money market (very low return, very low risk)
    """
    weights = xp.array([0.30, 0.25, 0.25, 0.20])   # Portfolio weights (must sum to 1.0)
    
    """
    Expected Annual Returns (μ):
    These represent the drift component in the GBM model.
    Values are expressed as decimals (e.g., 0.08 = 8% annual return).
    Based on historical market data and forward-looking assumptions.
    """
    mus_annual = xp.array([0.08, 0.05, 0.12, 0.03])  # Expected annual returns
    
    """
    Annual Volatilities (σ):
    These represent the diffusion component in the GBM model.
    Higher volatility = higher uncertainty in returns.
    Values based on historical standard deviations of asset returns.
    """
    vols_annual = xp.array([0.20, 0.15, 0.25, 0.10])  # Annual volatilities
    
    """
    Correlation Matrix (ρ):
    Captures the linear relationship between asset returns.
    - Diagonal elements = 1.0 (perfect self-correlation)
    - Off-diagonal elements between -1.0 and +1.0
    - Positive correlation: assets tend to move together
    - Negative correlation: assets tend to move opposite
    - Zero correlation: no linear relationship
    
    This matrix is crucial for portfolio diversification benefits.
    """
    corr = xp.array([
        [1.00, 0.40, 0.30, 0.10],  # Asset 1 correlations
        [0.40, 1.00, 0.35, 0.15],  # Asset 2 correlations  
        [0.30, 0.35, 1.00, 0.05],  # Asset 3 correlations
        [0.10, 0.15, 0.05, 1.00],  # Asset 4 correlations
    ])

    # =============================================================================
    # PORTFOLIO MATHEMATICS: COVARIANCE MATRIX CONSTRUCTION
    # =============================================================================
    
    """
    Convert correlation matrix to covariance matrix using the formula:
    Σ = D * ρ * D
    
    Where:
    - Σ (Sigma) is the covariance matrix
    - D is a diagonal matrix of volatilities
    - ρ (rho) is the correlation matrix
    
    This transformation is essential because:
    1. Correlations are scale-invariant (unitless)
    2. Covariances have units and are needed for portfolio variance calculation
    3. Portfolio variance = w^T * Σ * w (quadratic form)
    """
    D = xp.diag(vols_annual)  # Create diagonal matrix of volatilities
    Sigma = D @ corr @ D      # Matrix multiplication to get covariance matrix

    # =============================================================================
    # PORTFOLIO-LEVEL PARAMETERS
    # =============================================================================
    
    """
    Calculate aggregate portfolio parameters from individual asset parameters:
    
    1. Portfolio Expected Return (μ_p):
       μ_p = Σ(w_i * μ_i) = w^T * μ
       This is the weighted average of individual asset expected returns.
    
    2. Portfolio Variance (σ_p²):
       σ_p² = w^T * Σ * w
       This accounts for both individual asset variances and correlations.
    
    3. Portfolio Volatility (σ_p):
       σ_p = √(σ_p²)
       This is the portfolio standard deviation (risk measure).
    """
    mu_p = float(weights @ mus_annual)      # Portfolio expected return
    var_p = float(weights @ Sigma @ weights) # Portfolio variance
    sigma_p = math.sqrt(var_p)              # Portfolio volatility (standard deviation)

    # =============================================================================
    # RANDOM NUMBER GENERATION SETUP
    # =============================================================================
    
    """
    Set random seed for reproducible results.
    This ensures that the same sequence of random numbers is generated
    each time the simulation runs with the same parameters.
    Important for:
    - Debugging and testing
    - Comparing different portfolio configurations
    - Regulatory compliance and audit trails
    """
    cp.random.seed(seed) if on_gpu else _np.random.seed(seed)

    # =============================================================================
    # MONTE CARLO SIMULATION: GBM IMPLEMENTATION
    # =============================================================================
    
    print(f"\nStarting Monte Carlo simulation with {n_paths:,} paths...")
    t0 = time.time()  # Start timing the simulation
    
    """
    Geometric Brownian Motion (GBM) Analytical Solution:
    
    For a portfolio value S(t) following dS = μS dt + σS dW, the solution is:
    S(T) = S(0) * exp((μ - σ²/2) * T + σ * √T * Z)
    
    Where:
    - S(0) = initial portfolio value
    - T = time horizon
    - Z ~ N(0,1) = standard normal random variable
    - (μ - σ²/2) = drift term (Ito's correction for geometric process)
    - σ * √T = diffusion scaling factor
    
    The drift term includes the "Ito correction" (-σ²/2) which arises from
    the transformation from arithmetic to geometric Brownian motion.
    """
    
    # Pre-calculate deterministic components (same for all paths)
    drift_term = (mu_p - 0.5 * sigma_p**2) * horizon_years  # Drift with Ito correction
    diffusion_std = sigma_p * math.sqrt(horizon_years)       # Diffusion standard deviation
    
    """
    Generate Random Shocks:
    Create n_paths independent standard normal random variables.
    These represent the random market shocks that drive portfolio returns.
    Using float32 for memory efficiency with large simulations.
    """
    # Fix the dtype issue for NumPy compatibility
    if on_gpu:
        Z = xp.random.standard_normal(size=n_paths, dtype=xp.float32)
    else:
        Z = xp.random.standard_normal(size=n_paths).astype(xp.float32)
    
    """
    Calculate Terminal Portfolio Values:
    Apply the GBM formula to compute the portfolio value at the horizon
    for each of the n_paths scenarios.
    
    The exponential function ensures that portfolio values remain positive
    (a key advantage of GBM over arithmetic Brownian motion).
    """
    terminal_vals = xp.asarray(initial_value, dtype=xp.float32) * xp.exp(
        drift_term + diffusion_std * Z
    )

    # Ensure all GPU computations are complete before measuring time
    if on_gpu:
        cp.cuda.Stream.null.synchronize()

    sim_time = time.time() - t0  # Calculate simulation elapsed time

    # =============================================================================
    # RISK METRICS CALCULATION
    # =============================================================================
    
    print("Calculating risk metrics...")
    
    """
    Loss Distribution:
    Calculate losses as the difference between initial and terminal values.
    Positive values represent losses, negative values represent gains.
    This distribution is fundamental for risk metric calculation.
    """
    losses = initial_value - terminal_vals
    
    """
    Basic Statistical Measures:
    - Mean: Expected terminal portfolio value
    - Standard Deviation: Measure of portfolio value dispersion
    """
    mean_val = float(to_cpu(terminal_vals.mean()))
    std_val = float(to_cpu(terminal_vals.std()))

    """
    Value at Risk (VaR) at 95% Confidence Level:
    VaR answers: "What is the maximum loss we expect 95% of the time?"
    
    Calculation: 95th percentile of the loss distribution
    Interpretation: There's only a 5% chance of losing more than this amount
    
    VaR is widely used in:
    - Regulatory capital requirements (Basel III)
    - Risk budgeting and allocation
    - Performance measurement
    """
    var95 = float(to_cpu(percentile(losses, 95)))
    
    """
    Conditional Value at Risk (CVaR) at 95% Confidence Level:
    CVaR answers: "What is the average loss in the worst 5% of scenarios?"
    
    Also known as Expected Shortfall (ES), CVaR provides information about
    tail risk beyond what VaR captures. It's considered a more coherent
    risk measure than VaR because it satisfies all axioms of coherent risk measures.
    
    CVaR is particularly important for:
    - Understanding extreme loss scenarios
    - Capital allocation for tail risk
    - Stress testing and scenario analysis
    """
    tail_losses = losses[losses >= var95]  # Losses in the worst 5% of scenarios
    cvar95 = float(to_cpu(tail_losses.mean())) if len(tail_losses) > 0 else var95
    
    """
    Probability of Loss:
    Calculate the percentage of scenarios where the portfolio loses value.
    This gives intuitive insight into the likelihood of negative returns.
    """
    p_loss = float(to_cpu((terminal_vals < initial_value).mean()))

    """
    Percentile Analysis:
    Calculate key percentiles of the terminal value distribution:
    - 10th percentile: Pessimistic scenario (bottom 10% of outcomes)
    - 50th percentile: Median outcome (middle scenario)
    - 90th percentile: Optimistic scenario (top 10% of outcomes)
    
    These percentiles provide a comprehensive view of the return distribution
    and help in scenario planning and risk communication.
    """
    p10 = float(to_cpu(percentile(terminal_vals, 10)))   # Pessimistic case
    p50 = float(to_cpu(percentile(terminal_vals, 50)))   # Median case
    p90 = float(to_cpu(percentile(terminal_vals, 90)))   # Optimistic case

    # =============================================================================
    # RESULTS REPORTING
    # =============================================================================
    
    device = "GPU (CuPy/CUDA)" if on_gpu else "CPU (NumPy)"
    print("\n" + "="*60)
    print("MONTE CARLO PORTFOLIO RISK ANALYSIS")
    print("Geometric Brownian Motion Model - Terminal Value Analysis")
    print("="*60)
    
    print(f"\n📊 SIMULATION CONFIGURATION:")
    print(f"   Computing Device:     {device}")
    print(f"   Number of Scenarios:  {n_paths:,}")
    print(f"   Time Horizon:         {horizon_years} year(s)")
    print(f"   Initial Portfolio:    ${initial_value:,.2f}")
    print(f"   Portfolio μ (annual): {mu_p:.4f} ({mu_p*100:.2f}%)")
    print(f"   Portfolio σ (annual): {sigma_p:.4f} ({sigma_p*100:.2f}%)")
    
    print(f"\n📈 PORTFOLIO STATISTICS:")
    print(f"   Expected Terminal:    ${mean_val:,.2f}")
    print(f"   Standard Deviation:   ${std_val:,.2f}")
    print(f"   Expected Return:      {((mean_val/initial_value)-1)*100:.2f}%")
    
    print(f"\n📉 RISK METRICS:")
    print(f"   VaR (95% confidence): ${var95:,.2f}")
    print(f"   CVaR (95%):          ${cvar95:,.2f} (avg loss in worst 5%)")
    print(f"   Probability of Loss:  {p_loss*100:.2f}%")
    
    print(f"\n🎯 SCENARIO ANALYSIS:")
    print(f"   Pessimistic (10th %): ${p10:,.2f}")
    print(f"   Median (50th %):      ${p50:,.2f}")
    print(f"   Optimistic (90th %):  ${p90:,.2f}")
    
    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Simulation Time:      {sim_time:.3f} seconds")
    print(f"   Scenarios per Second: {n_paths/sim_time:,.0f}")
    print("="*60)
    
    return {
        'terminal_values': terminal_vals,
        'mean_val': mean_val,
        'std_val': std_val,
        'var95': var95,
        'cvar95': cvar95,
        'p_loss': p_loss,
        'percentiles': {'p10': p10, 'p50': p50, 'p90': p90},
        'simulation_time': sim_time
    }

if __name__ == "__main__":
    """
    Entry point for the Monte Carlo simulation.
    
    Environment Variables:
        N_PATHS: Number of simulation paths (default: 10,000,000)
        HORIZON_YEARS: Investment horizon in years (default: 1.0)
    
    Example usage:
        # Run with default parameters
        python main.py
        
        # Run with custom parameters via environment variables
        N_PATHS=1000000 HORIZON_YEARS=2.0 python main.py
    """
    n_paths = int(os.getenv("N_PATHS", "10000000"))
    horizon = float(os.getenv("HORIZON_YEARS", "1.0"))
    
    print("🚀 MONTE CARLO PORTFOLIO RISK SIMULATION")
    print("=" * 50)
    
    # Run the simulation
    results = main(n_paths=n_paths, horizon_years=horizon)
    
    print("\n✅ Simulation completed successfully!")
    print("📊 Use the returned results dictionary for further analysis.")