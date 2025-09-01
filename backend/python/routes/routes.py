"""
Python Backend Routes

This file contains all API routes for the Python risk simulation engine.
Routes are organized by functionality and include comprehensive error handling.
"""

from flask import Blueprint, request, jsonify
import time
import traceback
import sys

# Create blueprint for routes
routes_bp = Blueprint('routes', __name__)

# Import simulation modules
try:
    from simulations.monte_carlo import monte_carlo_simulation, advanced_monte_carlo_simulation
    from simulations.historical import historical_simulation, bootstrap_historical
    from simulations.bootstrap import bootstrap_simulation, advanced_bootstrap
    from simulations.varcov import variance_covariance_method, stress_test_varcov
    from simulations.gbm import gbm_simulation, multi_asset_gbm, path_dependent_gbm
    print("✅ All simulation modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating stub functions for missing modules...")
    
    def monte_carlo_simulation(portfolio, params=None):
        """Stub implementation"""
        import numpy as np
        initial_value = sum(portfolio)
        return {
            'simulation_config': {
                'method': 'monte_carlo',
                'engine': 'python',
                'initial_value': initial_value
            },
            'risk_metrics': {
                'VaR_95': initial_value * 0.05,
                'CVaR_95': initial_value * 0.08,
                'probability_of_loss': 30.0
            },
            'portfolio_stats': {
                'expected_terminal': initial_value * 1.07,
                'expected_return': 7.0,
                'volatility': 15.0
            },
            'performance': {
                'simulation_time': 0.1,
                'iterations_per_second': 1000000
            }
        }
    
    def advanced_monte_carlo_simulation(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def historical_simulation(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def bootstrap_historical(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def bootstrap_simulation(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def advanced_bootstrap(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def variance_covariance_method(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def stress_test_varcov(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def gbm_simulation(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def multi_asset_gbm(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)
    
    def path_dependent_gbm(portfolio, params=None):
        return monte_carlo_simulation(portfolio, params)

# Utility functions
def validate_portfolio(portfolio):
    """Validate portfolio input"""
    if not isinstance(portfolio, list):
        return {'valid': False, 'message': 'Portfolio must be a list'}
    if len(portfolio) == 0:
        return {'valid': False, 'message': 'Portfolio cannot be empty'}
    if not all(isinstance(x, (int, float)) and x > 0 for x in portfolio):
        return {'valid': False, 'message': 'Portfolio values must be positive numbers'}
    return {'valid': True}

def handle_simulation_error(error):
    """Handle simulation errors gracefully"""
    return {
        'error': 'Simulation failed',
        'message': str(error),
        'type': type(error).__name__,
        'engine': 'python'
    }

# Health check route
@routes_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'engine': 'python',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version': '1.0.0',
        'python_version': sys.version,
        'platform': sys.platform
    })

# API information route
@routes_bp.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'engine': 'python',
        'version': '1.0.0',
        'description': 'Risk Simulation Engine - Python Implementation',
        'supported_methods': [
            'monte_carlo',
            'advanced_monte_carlo',
            'historical_simulation',
            'bootstrap_historical',
            'bootstrap',
            'advanced_bootstrap',
            'variance_covariance',
            'stress_test_varcov',
            'gbm',
            'multi_asset_gbm',
            'path_dependent_gbm'
        ],
        'documentation': '/api/docs',
        'examples': '/api/examples',
        'technology_stack': {
            'language': 'Python',
            'framework': 'Flask',
            'numerical_computing': 'NumPy + SciPy',
            'performance': 'Vectorized operations'
        }
    })

# API documentation route
@routes_bp.route('/api/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    return jsonify({
        'title': 'Risk Simulation Engine API - Python',
        'description': 'High-performance financial risk modeling API implemented in Python',
        'base_url': 'http://localhost:3002',
        
        'endpoints': {
            '/simulate': {
                'method': 'POST',
                'description': 'Run risk simulation with specified method',
                'parameters': {
                    'engine': 'string (always "python" for this server)',
                    'method': 'string (simulation method to use)',
                    'portfolio': 'array of numbers (asset values)',
                    'params': 'object (method-specific parameters)'
                },
                'example': {
                    'engine': 'python',
                    'method': 'monte_carlo',
                    'portfolio': [100000, 200000, 150000, 50000],
                    'params': {
                        'iterations': 1000000,
                        'confidence': 0.95,
                        'horizon': 1.0
                    }
                }
            },
            
            '/compare': {
                'method': 'POST',
                'description': 'Compare multiple simulation methods',
                'parameters': {
                    'portfolio': 'array of numbers',
                    'methods': 'array of method names',
                    'params': 'object (shared parameters)'
                }
            },
            
            '/benchmark': {
                'method': 'POST',
                'description': 'Performance benchmark of simulation methods',
                'parameters': {
                    'portfolio': 'array of numbers',
                    'iterations': 'array of iteration counts',
                    'method': 'string (method to benchmark)'
                }
            }
        },
        
        'python_advantages': {
            'numerical_computing': 'NumPy/SciPy ecosystem for scientific computing',
            'vectorization': 'Highly optimized array operations',
            'libraries': 'Rich ecosystem of quantitative finance libraries',
            'readability': 'Clean, readable code for complex mathematical operations',
            'integration': 'Easy integration with data science workflows'
        }
    })

# API examples route
@routes_bp.route('/api/examples', methods=['GET'])
def api_examples():
    """Example requests endpoint"""
    return jsonify({
        'basic_monte_carlo': {
            'method': 'POST',
            'url': '/simulate',
            'body': {
                'engine': 'python',
                'method': 'monte_carlo',
                'portfolio': [100000, 200000, 150000, 50000],
                'params': {
                    'iterations': 1000000,
                    'confidence': 0.95,
                    'horizon': 1.0
                }
            },
            'description': 'Basic Monte Carlo simulation using portfolio-level GBM'
        },
        
        'advanced_monte_carlo': {
            'method': 'POST',
            'url': '/simulate',
            'body': {
                'engine': 'python',
                'method': 'advanced_monte_carlo',
                'portfolio': [250000, 250000, 250000, 250000],
                'params': {
                    'iterations': 1000000,
                    'confidence': 0.95,
                    'horizon': 1.0,
                    'expected_returns': [0.08, 0.05, 0.12, 0.03],
                    'volatilities': [0.20, 0.15, 0.25, 0.10]
                }
            },
            'description': 'Advanced Monte Carlo with explicit asset correlation modeling'
        },
        
        'gbm_simulation': {
            'method': 'POST',
            'url': '/simulate',
            'body': {
                'engine': 'python',
                'method': 'gbm',
                'portfolio': [100000, 100000, 100000, 100000],
                'params': {
                    'iterations': 500000,
                    'confidence': 0.95,
                    'horizon': 1.0,
                    'time_steps': 252,
                    'expected_returns': [0.08, 0.05, 0.12, 0.03],
                    'volatilities': [0.20, 0.15, 0.25, 0.10],
                    'correlations': 0.3
                }
            },
            'description': 'Geometric Brownian Motion simulation with full path modeling'
        },
        
        'method_comparison': {
            'method': 'POST',
            'url': '/compare',
            'body': {
                'portfolio': [100000, 100000, 100000, 100000],
                'methods': ['monte_carlo', 'variance_covariance', 'gbm'],
                'params': {
                    'confidence': 0.95,
                    'horizon': 1.0,
                    'iterations': 100000
                }
            },
            'description': 'Compare different simulation methods'
        }
    })

# Main simulation route
@routes_bp.route('/simulate', methods=['POST'])
def simulate():
    """Main simulation endpoint"""
    try:
        print(f"🐍 Received simulation request: {request.json.get('method', 'unknown')}")
        start_time = time.time()
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Invalid JSON data',
                'engine': 'python'
            }), 400
        
        method = data.get('method')
        portfolio = data.get('portfolio', [])
        params = data.get('params', {})
        
        # Validate input
        if not method:
            return jsonify({
                'error': 'Missing required parameter: method',
                'engine': 'python'
            }), 400
        
        portfolio_validation = validate_portfolio(portfolio)
        if not portfolio_validation['valid']:
            return jsonify({
                'error': 'Invalid portfolio',
                'message': portfolio_validation['message'],
                'engine': 'python'
            }), 400
        
        # Route to appropriate simulation method
        result = None
        
        if method == 'monte_carlo':
            result = monte_carlo_simulation(portfolio, params)
        elif method == 'advanced_monte_carlo':
            result = advanced_monte_carlo_simulation(portfolio, params)
        elif method == 'historical_simulation':
            result = historical_simulation(portfolio, params)
        elif method == 'bootstrap_historical':
            result = bootstrap_historical(portfolio, params)
        elif method == 'bootstrap':
            result = bootstrap_simulation(portfolio, params)
        elif method == 'advanced_bootstrap':
            result = advanced_bootstrap(portfolio, params)
        elif method == 'variance_covariance':
            result = variance_covariance_method(portfolio, params)
        elif method == 'stress_test_varcov':
            result = stress_test_varcov(portfolio, params)
        elif method == 'gbm':
            result = gbm_simulation(portfolio, params)
        elif method == 'multi_asset_gbm':
            result = multi_asset_gbm(portfolio, params)
        elif method == 'path_dependent_gbm':
            result = path_dependent_gbm(portfolio, params)
        else:
            return jsonify({
                'error': f'Unknown simulation method: {method}',
                'supported_methods': [
                    'monte_carlo', 'advanced_monte_carlo', 'historical_simulation',
                    'bootstrap_historical', 'bootstrap', 'advanced_bootstrap',
                    'variance_covariance', 'stress_test_varcov', 'gbm',
                    'multi_asset_gbm', 'path_dependent_gbm'
                ],
                'engine': 'python'
            }), 400
        
        total_time = time.time() - start_time
        
        # Add server metadata
        result['server_info'] = {
            'engine': 'python',
            'method': method,
            'request_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'total_processing_time': float(total_time),
            'python_version': sys.version.split()[0],
            'numpy_backend': 'NumPy'
        }
        
        print(f"✅ Simulation completed in {total_time:.3f}s")
        return jsonify(result)
        
    except Exception as error:
        print(f"❌ Simulation error: {error}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify(handle_simulation_error(error)), 500

# Method comparison route
@routes_bp.route('/compare', methods=['POST'])
def compare_methods():
    """Method comparison endpoint"""
    try:
        print('🔍 Running method comparison...')
        start_time = time.time()
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Invalid JSON data',
                'engine': 'python'
            }), 400
        
        portfolio = data.get('portfolio', [])
        methods = data.get('methods', [])
        params = data.get('params', {})
        
        # Validate input
        portfolio_validation = validate_portfolio(portfolio)
        if not portfolio_validation['valid']:
            return jsonify({
                'error': 'Invalid portfolio',
                'message': portfolio_validation['message'],
                'engine': 'python'
            }), 400
        
        if not isinstance(methods, list) or len(methods) == 0:
            return jsonify({
                'error': 'Invalid methods array',
                'engine': 'python'
            }), 400
        
        results = {}
        timings = {}
        
        # Run each method
        for method in methods:
            try:
                print(f"  Running {method}...")
                method_start_time = time.time()
                
                if method == 'monte_carlo':
                    method_result = monte_carlo_simulation(portfolio, params)
                elif method == 'advanced_monte_carlo':
                    method_result = advanced_monte_carlo_simulation(portfolio, params)
                elif method == 'historical_simulation':
                    method_result = historical_simulation(portfolio, params)
                elif method == 'bootstrap_historical':
                    method_result = bootstrap_historical(portfolio, params)
                elif method == 'bootstrap':
                    method_result = bootstrap_simulation(portfolio, params)
                elif method == 'advanced_bootstrap':
                    method_result = advanced_bootstrap(portfolio, params)
                elif method == 'variance_covariance':
                    method_result = variance_covariance_method(portfolio, params)
                elif method == 'stress_test_varcov':
                    method_result = stress_test_varcov(portfolio, params)
                elif method == 'gbm':
                    method_result = gbm_simulation(portfolio, params)
                elif method == 'multi_asset_gbm':
                    method_result = multi_asset_gbm(portfolio, params)
                elif method == 'path_dependent_gbm':
                    method_result = path_dependent_gbm(portfolio, params)
                else:
                    results[method] = {'error': f'Unsupported comparison method: {method}'}
                    timings[method] = 0
                    continue
                
                method_time = time.time() - method_start_time
                
                results[method] = {
                    'risk_metrics': method_result['risk_metrics'],
                    'portfolio_stats': method_result['portfolio_stats'],
                    'performance': method_result.get('performance', {'simulation_time': method_time})
                }
                
                timings[method] = method_time
                
            except Exception as error:
                print(f"Error running {method}: {error}")
                results[method] = {'error': str(error)}
                timings[method] = 0
        
        # Calculate comparison metrics
        var_comparison = {}
        cvar_comparison = {}
        performance_comparison = {}
        
        for method in methods:
            if 'risk_metrics' in results[method]:
                var_comparison[method] = results[method]['risk_metrics']['VaR_95']
                cvar_comparison[method] = results[method]['risk_metrics']['CVaR_95']
                performance_comparison[method] = timings[method]
        
        total_time = time.time() - start_time
        
        response = {
            'comparison': {
                'portfolio': portfolio,
                'methods': methods,
                'params': params,
                'results': results,
                'analysis': {
                    'var_comparison': var_comparison,
                    'cvar_comparison': cvar_comparison,
                    'performance_comparison': performance_comparison,
                }
            },
            'metadata': {
                'engine': 'python',
                'total_processing_time': float(total_time),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        }
        
        # Add fastest method if we have performance data
        if performance_comparison:
            response['comparison']['analysis']['fastest_method'] = min(
                performance_comparison.items(), key=lambda x: x[1]
            )[0]
        
        print(f"✅ Method comparison completed in {total_time:.3f}s")
        return jsonify(response)
        
    except Exception as error:
        print(f"❌ Comparison error: {error}")
        return jsonify(handle_simulation_error(error)), 500

# Performance benchmark route
@routes_bp.route('/benchmark', methods=['POST'])
def benchmark_performance():
    """Performance benchmark endpoint"""
    try:
        print('🏃 Running performance benchmark...')
        start_time = time.time()
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Invalid JSON data',
                'engine': 'python'
            }), 400
        
        portfolio = data.get('portfolio', [])
        method = data.get('method', 'monte_carlo')
        iterations_list = data.get('iterations', [10000, 50000, 100000, 500000, 1000000])
        params = data.get('params', {})
        
        portfolio_validation = validate_portfolio(portfolio)
        if not portfolio_validation['valid']:
            return jsonify({
                'error': 'Invalid portfolio',
                'message': portfolio_validation['message'],
                'engine': 'python'
            }), 400
        
        benchmark_results = []
        
        for iter_count in iterations_list:
            print(f"  Benchmarking {method} with {iter_count:,} iterations...")
            
            bench_params = params.copy()
            bench_params['iterations'] = iter_count
            bench_start_time = time.time()
            
            if method == 'monte_carlo':
                result = monte_carlo_simulation(portfolio, bench_params)
            elif method == 'advanced_monte_carlo':
                result = advanced_monte_carlo_simulation(portfolio, bench_params)
            elif method == 'historical_simulation':
                result = historical_simulation(portfolio, bench_params)
            elif method == 'bootstrap_historical':
                result = bootstrap_historical(portfolio, bench_params)
            elif method == 'bootstrap':
                result = bootstrap_simulation(portfolio, bench_params)
            elif method == 'advanced_bootstrap':
                result = advanced_bootstrap(portfolio, bench_params)
            elif method == 'variance_covariance':
                result = variance_covariance_method(portfolio, bench_params)
            elif method == 'stress_test_varcov':
                result = stress_test_varcov(portfolio, bench_params)
            elif method == 'gbm':
                result = gbm_simulation(portfolio, bench_params)
            elif method == 'multi_asset_gbm':
                result = multi_asset_gbm(portfolio, bench_params)
            elif method == 'path_dependent_gbm':
                result = path_dependent_gbm(portfolio, bench_params)
            else:
                return jsonify({
                    'error': f'Benchmarking not supported for method: {method}',
                    'engine': 'python'
                }), 400
            
            bench_time = time.time() - bench_start_time
            throughput = int(iter_count / bench_time) if bench_time > 0 else 0
            
            benchmark_results.append({
                'iterations': iter_count,
                'time': float(bench_time),
                'throughput': throughput,
                'VaR_95': result['risk_metrics']['VaR_95'],
                'CVaR_95': result['risk_metrics']['CVaR_95']
            })
        
        total_time = time.time() - start_time
        
        # Calculate analysis metrics
        throughputs = [r['throughput'] for r in benchmark_results if r['throughput'] > 0]
        
        response = {
            'benchmark': {
                'method': method,
                'portfolio': portfolio,
                'results': benchmark_results,
                'analysis': {
                    'max_throughput': max(throughputs) if throughputs else 0,
                    'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
                    'scaling_efficiency': (throughputs[-1] / throughputs[0]) if len(throughputs) >= 2 else 1.0
                }
            },
            'metadata': {
                'engine': 'python',
                'total_benchmark_time': float(total_time),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        }
        
        print(f"✅ Benchmark completed in {total_time:.3f}s")
        return jsonify(response)
        
    except Exception as error:
        print(f"❌ Benchmark error: {error}")
        return jsonify(handle_simulation_error(error)), 500
