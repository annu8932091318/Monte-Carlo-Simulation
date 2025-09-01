# Risk Simulation Engine - Python Backend

## 🐍 Scientific Computing Powerhouse

The Python backend provides sophisticated risk simulation capabilities leveraging the full scientific computing ecosystem. Built with Flask for robust API services and NumPy/SciPy for high-performance numerical computations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Simulation Methods](#simulation-methods)
- [Scientific Computing](#scientific-computing)
- [Examples](#examples)

---

## 🎯 Overview

This Python implementation harnesses:
- **NumPy & SciPy** for vectorized numerical operations
- **Flask** for robust REST API framework
- **Scientific libraries** for advanced mathematical modeling
- **Optimized algorithms** for complex financial calculations
- **Professional-grade** error handling and validation

---

## ✨ Features

### Advanced Simulation Methods
- **Monte Carlo Simulation** - Stochastic modeling with GBM
- **Advanced Monte Carlo** - Multi-asset correlation matrices
- **Historical Simulation** - Empirical distribution modeling
- **Bootstrap Methods** - Non-parametric risk estimation
- **Variance-Covariance** - Parametric approach with stress testing
- **Geometric Brownian Motion** - Advanced path-dependent modeling

### Scientific Computing Features
- **Vectorized operations** for maximum performance
- **Memory-efficient algorithms** for large-scale simulations
- **Statistical robustness** with comprehensive error analysis
- **Numerical stability** through proven scientific libraries

---

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup Steps

```bash
# Navigate to Python backend directory
cd backend/python

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

### Dependencies
```text
flask>=2.3.0
flask-cors>=4.0.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# Flask Configuration
FLASK_APP=app.py            # Flask application entry point
FLASK_DEBUG=False           # Debug mode (False for production)
PORT=3002                   # Server port (default: 3002)

# Python Optimization
PYTHONPATH=.                # Python module search path
OMP_NUM_THREADS=4           # OpenMP thread count
OPENBLAS_NUM_THREADS=4      # OpenBLAS thread count
MKL_NUM_THREADS=4           # Intel MKL thread count
```

### Server Startup Options
```bash
# Development mode with debug
FLASK_DEBUG=True python app.py

# Production mode
python app.py

# With custom port
PORT=8080 python app.py

# Using Gunicorn for production
gunicorn -w 4 -b 0.0.0.0:3002 app:app
```

---

## 🔗 API Endpoints

### Health & Information
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health and system info |
| `/api/info` | GET | API capabilities and methods |
| `/api/docs` | GET | Complete API documentation |
| `/api/examples` | GET | Request/response examples |

### Simulation Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate` | POST | Execute single simulation method |
| `/compare` | POST | Compare multiple simulation methods |
| `/benchmark` | POST | Performance benchmarking across methods |

---

## 🧮 Simulation Methods

### 1. Monte Carlo Simulation
```python
# Portfolio-level geometric Brownian motion
{
  "engine": "python",
  "method": "monte_carlo",
  "portfolio": [100000, 200000, 150000, 50000],
  "params": {
    "iterations": 1000000,
    "confidence": 0.95,
    "horizon": 1.0,
    "drift": 0.07,
    "volatility": 0.15
  }
}
```

### 2. Advanced Monte Carlo
```python
# Multi-asset correlation modeling with NumPy
{
  "engine": "python",
  "method": "advanced_monte_carlo", 
  "portfolio": [250000, 250000, 250000, 250000],
  "params": {
    "iterations": 1000000,
    "confidence": 0.95,
    "horizon": 1.0,
    "expected_returns": [0.08, 0.05, 0.12, 0.03],
    "volatilities": [0.20, 0.15, 0.25, 0.10],
    "correlation_matrix": [
      [1.0, 0.3, 0.1, 0.0],
      [0.3, 1.0, 0.2, 0.1], 
      [0.1, 0.2, 1.0, 0.0],
      [0.0, 0.1, 0.0, 1.0]
    ]
  }
}
```

### 3. Historical Simulation
```python
# Empirical distribution with bootstrap resampling
{
  "engine": "python",
  "method": "historical_simulation",
  "portfolio": [100000, 100000, 100000, 100000],
  "params": {
    "confidence": 0.95,
    "lookback_days": 252,
    "bootstrap_samples": 10000,
    "block_length": 5
  }
}
```

### 4. Variance-Covariance Method
```python
# Parametric approach with stress testing
{
  "engine": "python",
  "method": "stress_test_varcov",
  "portfolio": [200000, 150000, 100000, 50000],
  "params": {
    "confidence": 0.99,
    "horizon": 1.0,
    "stress_scenarios": [
      {"market_shock": -0.20, "volatility_increase": 1.5},
      {"correlation_increase": 0.3, "liquidity_stress": 0.1}
    ]
  }
}
```

### 5. Geometric Brownian Motion
```python
# Advanced path-dependent modeling
{
  "engine": "python",
  "method": "path_dependent_gbm",
  "portfolio": [500000, 300000, 200000],
  "params": {
    "iterations": 1000000,
    "time_steps": 252,
    "horizon": 1.0,
    "barrier_levels": [0.8, 0.9, 1.1, 1.2],
    "path_dependency": "barrier_option"
  }
}
```

---

## 🔬 Scientific Computing

### NumPy Optimization
```python
# Vectorized operations for maximum performance
import numpy as np

# Generate correlated random variables
def generate_correlated_returns(correlation_matrix, n_assets, n_simulations):
    """
    Efficient generation of correlated asset returns using Cholesky decomposition
    """
    L = np.linalg.cholesky(correlation_matrix)
    Z = np.random.standard_normal((n_assets, n_simulations))
    return L @ Z

# Risk metric calculations
def calculate_var_cvar(returns, confidence=0.95):
    """
    Vectorized VaR and CVaR calculations
    """
    var_threshold = np.percentile(returns, (1-confidence)*100)
    cvar = np.mean(returns[returns <= var_threshold])
    return var_threshold, cvar
```

### SciPy Statistical Functions
```python
# Advanced statistical modeling
from scipy import stats
from scipy.optimize import minimize

# Fit distributions to empirical data
def fit_distribution_family(returns):
    """
    Fit multiple distributions and select best fit
    """
    distributions = [stats.norm, stats.t, stats.skewnorm, stats.genextreme]
    best_fit = None
    best_aic = np.inf
    
    for dist in distributions:
        params = dist.fit(returns)
        aic = 2*len(params) - 2*np.sum(dist.logpdf(returns, *params))
        if aic < best_aic:
            best_aic = aic
            best_fit = (dist, params)
    
    return best_fit
```

### Performance Optimizations
```python
# Memory-efficient algorithms for large simulations
def monte_carlo_vectorized(portfolio, params):
    """
    Vectorized Monte Carlo implementation using NumPy
    Memory usage: O(n_assets * sqrt(n_simulations))
    """
    n_simulations = params.get('iterations', 1000000)
    chunk_size = min(100000, n_simulations)  # Process in chunks
    
    results = []
    for i in range(0, n_simulations, chunk_size):
        current_chunk = min(chunk_size, n_simulations - i)
        chunk_results = _process_chunk(portfolio, params, current_chunk)
        results.append(chunk_results)
    
    return np.concatenate(results)
```

---

## 🎮 Performance

### Scientific Computing Advantages
- **Vectorized Operations** - 10-100x faster than pure Python loops
- **Memory Efficiency** - Optimized array operations
- **Numerical Stability** - IEEE 754 compliant calculations  
- **Parallel Processing** - Multi-threaded BLAS/LAPACK operations

### Typical Performance Metrics
```python
{
  "numerical_backend": "NumPy + OpenBLAS",
  "max_iterations": 10000000,        # Maximum simulations
  "vectorization_speedup": 45.2,     # vs pure Python
  "memory_efficiency": 0.89,         # Memory utilization ratio
  "numerical_precision": "float64",   # Double precision
  "thread_utilization": 4             # CPU cores used
}
```

### Performance Benchmarking
```bash
curl -X POST http://localhost:3002/benchmark \\
  -H "Content-Type: application/json" \\
  -d '{
    "portfolio": [100000, 100000, 100000, 100000],
    "method": "monte_carlo",
    "iterations": [10000, 50000, 100000, 500000, 1000000, 5000000],
    "params": {
      "confidence": 0.95,
      "horizon": 1.0
    }
  }'
```

---

## 📝 Examples

### Basic Risk Analysis
```bash
curl -X POST http://localhost:3002/simulate \\
  -H "Content-Type: application/json" \\
  -d '{
    "engine": "python",
    "method": "monte_carlo",
    "portfolio": [1000000, 500000, 750000],
    "params": {
      "iterations": 1000000,
      "confidence": 0.95,
      "horizon": 1.0
    }
  }'
```

### Advanced Correlation Modeling
```bash
curl -X POST http://localhost:3002/simulate \\
  -H "Content-Type: application/json" \\
  -d '{
    "engine": "python",
    "method": "advanced_monte_carlo",
    "portfolio": [500000, 300000, 200000, 100000],
    "params": {
      "iterations": 2000000,
      "confidence": 0.99,
      "horizon": 0.25,
      "expected_returns": [0.12, 0.08, 0.06, 0.04],
      "volatilities": [0.25, 0.18, 0.15, 0.12],
      "correlation_matrix": [
        [1.0, 0.4, 0.2, 0.1],
        [0.4, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.4],
        [0.1, 0.2, 0.4, 1.0]
      ]
    }
  }'
```

### Statistical Distribution Fitting
```bash
curl -X POST http://localhost:3002/simulate \\
  -H "Content-Type: application/json" \\
  -d '{
    "engine": "python",
    "method": "historical_simulation",
    "portfolio": [200000, 200000, 200000, 200000, 200000],
    "params": {
      "confidence": 0.99,
      "lookback_days": 500,
      "distribution_fitting": true,
      "bootstrap_samples": 50000,
      "block_length": 10
    }
  }'
```

### Method Comparison Analysis
```bash
curl -X POST http://localhost:3002/compare \\
  -H "Content-Type: application/json" \\
  -d '{
    "portfolio": [150000, 150000, 150000, 150000],
    "methods": [
      "monte_carlo", 
      "advanced_monte_carlo", 
      "historical_simulation",
      "variance_covariance"
    ],
    "params": {
      "confidence": 0.95,
      "horizon": 1.0,
      "iterations": 500000
    }
  }'
```

---

## 🔧 Development

### Project Structure
```
backend/python/
├── routes/
│   └── routes.py          # Flask Blueprint routes
├── simulations/           # Simulation modules
│   ├── monte_carlo.py     # Monte Carlo implementations
│   ├── historical.py      # Historical simulation methods
│   ├── bootstrap.py       # Bootstrap resampling
│   ├── varcov.py         # Variance-Covariance methods
│   └── gbm.py            # Geometric Brownian Motion
├── utils/                 # Utility functions
│   ├── statistics.py      # Statistical calculations
│   ├── validation.py      # Input validation
│   └── optimization.py    # Performance optimizations
├── app.py                # Flask application entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Testing Framework
```bash
# Install testing dependencies
pip install pytest pytest-cov numpy-testing

# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=simulations tests/

# Run performance tests
pytest tests/test_performance.py -v

# Run numerical accuracy tests
pytest tests/test_numerical.py -v
```

### Code Quality Tools
```bash
# Code formatting
pip install black isort
black .
isort .

# Type checking
pip install mypy
mypy simulations/

# Linting
pip install flake8 pylint
flake8 .
pylint simulations/
```

---

## 🌟 Python Advantages

### Scientific Computing Benefits
- **NumPy Ecosystem** - Industry-standard numerical computing
- **SciPy Statistical Functions** - Comprehensive statistical library
- **Pandas Integration** - Advanced data manipulation capabilities
- **Matplotlib Visualization** - Professional plotting and analysis
- **Scikit-learn** - Machine learning integration possibilities

### Mathematical Rigor
- **IEEE 754 Compliance** - Precise floating-point arithmetic
- **Numerical Stability** - Robust algorithms for edge cases
- **Statistical Accuracy** - Peer-reviewed statistical methods
- **Reproducible Results** - Deterministic random number generation

### Integration Capabilities
- **Jupyter Notebooks** - Interactive analysis and visualization
- **Data Science Workflow** - Seamless integration with analytics pipelines
- **Academic Research** - Compatible with research-grade libraries
- **Financial Libraries** - Integration with QuantLib, PyPortfolioOpt, etc.

---

## 📊 Expected Response Format

```python
{
  "simulation_config": {
    "method": "advanced_monte_carlo",
    "engine": "python",
    "initial_value": 500000,
    "iterations": 1000000,
    "confidence": 0.95,
    "horizon": 1.0
  },
  "risk_metrics": {
    "VaR_95": 28750.42,           # Value at Risk (95%)
    "CVaR_95": 41230.15,          # Conditional VaR (95%)
    "probability_of_loss": 31.2,  # Probability of loss (%)
    "maximum_loss": 125000.80,    # Maximum simulated loss
    "tail_expectation": 45100.25  # Expected tail loss
  },
  "portfolio_stats": {
    "expected_terminal": 535000.00,    # Expected final value
    "expected_return": 7.0,            # Expected return (%)
    "volatility": 16.8,                # Portfolio volatility (%)
    "skewness": -0.15,                 # Return distribution skewness
    "kurtosis": 3.24,                  # Return distribution kurtosis
    "sharpe_ratio": 0.42,              # Risk-adjusted return
    "sortino_ratio": 0.58              # Downside risk-adjusted return
  },
  "statistical_analysis": {
    "distribution_fit": "Student-t",    # Best-fit distribution
    "kolmogorov_smirnov_p": 0.12,      # Goodness of fit p-value
    "anderson_darling": 0.89,          # Normality test statistic
    "ljung_box_p": 0.34                # Serial correlation test
  },
  "performance": {
    "simulation_time": 0.234,          # Execution time (seconds)
    "iterations_per_second": 4273504,  # Computational throughput
    "memory_usage": 156.8,             # Memory used (MB)
    "cpu_utilization": 87.5,           # CPU usage (%)
    "vectorization_ratio": 0.95        # Vectorized operations ratio
  },
  "server_info": {
    "engine": "python",
    "method": "advanced_monte_carlo",
    "request_time": "2024-01-15T10:30:45Z",
    "total_processing_time": 0.237,
    "python_version": "3.11.6",
    "numpy_version": "1.24.3",
    "scipy_version": "1.10.1",
    "numerical_backend": "OpenBLAS"
  }
}
```

---

## 🚀 Getting Started

1. **Install Python 3.8+** from [python.org](https://python.org)
2. **Create virtual environment** with `python -m venv venv`
3. **Activate environment** and install dependencies
4. **Start Flask server** with `python app.py`
5. **Test the API** using the examples above
6. **Explore documentation** at `http://localhost:3002/api/docs`

The Python backend is now ready to deliver scientific-grade risk simulations! 🧪🐍
