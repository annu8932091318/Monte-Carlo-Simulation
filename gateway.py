"""
Multi-Language Risk Simulation API Gateway

This gateway provides a unified interface to both Node.js and Python
risk simulation engines, allowing users to leverage the strengths
of each language ecosystem for financial risk modeling.

Key Features:
- Automatic engine routing based on preferences
- Load balancing between engines
- Unified response format
- Performance monitoring
- Fallback mechanisms
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import threading
import os
import sys
from typing import Dict, Any, Optional, List

app = Flask(__name__)
CORS(app)

# Configuration
NODE_SERVER_URL = os.getenv('NODE_SERVER_URL', 'http://localhost:3001')
PYTHON_SERVER_URL = os.getenv('PYTHON_SERVER_URL', 'http://localhost:3002')
GATEWAY_PORT = int(os.getenv('GATEWAY_PORT', 3000))

# Health monitoring
engine_health = {
    'node': {'status': 'unknown', 'last_check': 0, 'response_time': 0},
    'python': {'status': 'unknown', 'last_check': 0, 'response_time': 0}
}

def check_engine_health(engine: str, url: str) -> Dict[str, Any]:
    """Check if an engine is healthy"""
    try:
        start_time = time.time()
        response = requests.get(f"{url}/health", timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return {
                'status': 'healthy',
                'response_time': response_time,
                'last_check': time.time()
            }
        else:
            return {
                'status': 'unhealthy',
                'response_time': float('inf'),
                'last_check': time.time()
            }
    except Exception as e:
        return {
            'status': 'unreachable',
            'response_time': float('inf'),
            'last_check': time.time(),
            'error': str(e)
        }

def update_engine_health():
    """Background thread to monitor engine health"""
    while True:
        engine_health['node'] = check_engine_health('node', NODE_SERVER_URL)
        engine_health['python'] = check_engine_health('python', PYTHON_SERVER_URL)
        time.sleep(30)  # Check every 30 seconds

# Start health monitoring thread
health_thread = threading.Thread(target=update_engine_health, daemon=True)
health_thread.start()

def select_engine(requested_engine: Optional[str], method: str) -> str:
    """
    Select the best engine for the request based on:
    1. User preference
    2. Method compatibility
    3. Engine health and performance
    """
    
    # Method-engine preferences
    node_preferred_methods = [
        'monte_carlo', 'historical_simulation', 'bootstrap', 'variance_covariance'
    ]
    
    python_preferred_methods = [
        'advanced_monte_carlo', 'advanced_bootstrap', 'gbm', 'multi_asset_gbm',
        'path_dependent_gbm', 'stress_test_varcov'
    ]
    
    # If user specified an engine, prefer it (if healthy)
    if requested_engine:
        if requested_engine in engine_health:
            if engine_health[requested_engine]['status'] == 'healthy':
                return requested_engine
            else:
                print(f"⚠️  Requested engine {requested_engine} is not healthy, falling back...")
    
    # Method-based selection
    if method in python_preferred_methods:
        if engine_health['python']['status'] == 'healthy':
            return 'python'
        elif engine_health['node']['status'] == 'healthy':
            print(f"🔄 Python preferred for {method} but unhealthy, using Node.js")
            return 'node'
    
    if method in node_preferred_methods:
        if engine_health['node']['status'] == 'healthy':
            return 'node'
        elif engine_health['python']['status'] == 'healthy':
            print(f"🔄 Node.js preferred for {method} but unhealthy, using Python")
            return 'python'
    
    # Performance-based selection (prefer faster engine)
    node_healthy = engine_health['node']['status'] == 'healthy'
    python_healthy = engine_health['python']['status'] == 'healthy'
    
    if node_healthy and python_healthy:
        # Both healthy, choose faster one
        if engine_health['node']['response_time'] < engine_health['python']['response_time']:
            return 'node'
        else:
            return 'python'
    elif node_healthy:
        return 'node'
    elif python_healthy:
        return 'python'
    else:
        # Both unhealthy, try node first
        return 'node'

def forward_request(engine: str, endpoint: str, data: Dict[str, Any] = None, method: str = 'POST') -> Dict[str, Any]:
    """Forward request to the selected engine"""
    
    url_map = {
        'node': NODE_SERVER_URL,
        'python': PYTHON_SERVER_URL
    }
    
    base_url = url_map.get(engine)
    if not base_url:
        return {
            'error': f'Unknown engine: {engine}',
            'available_engines': list(url_map.keys())
        }
    
    try:
        start_time = time.time()
        
        if method == 'GET':
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
        else:  # POST
            response = requests.post(f"{base_url}{endpoint}", json=data, timeout=300)
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Add gateway metadata
            result['gateway_info'] = {
                'selected_engine': engine,
                'gateway_processing_time': float(processing_time),
                'engine_selection_reason': 'automatic_selection',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
            
            return result
        else:
            return {
                'error': f'Engine {engine} returned status {response.status_code}',
                'engine_response': response.text,
                'gateway_info': {
                    'selected_engine': engine,
                    'error_occurred': True
                }
            }
    
    except requests.exceptions.Timeout:
        return {
            'error': f'Request to {engine} engine timed out',
            'gateway_info': {
                'selected_engine': engine,
                'timeout_occurred': True
            }
        }
    except Exception as e:
        return {
            'error': f'Failed to communicate with {engine} engine: {str(e)}',
            'gateway_info': {
                'selected_engine': engine,
                'communication_error': True
            }
        }

@app.route('/health', methods=['GET'])
def gateway_health():
    """Gateway health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'Risk Simulation API Gateway',
        'version': '1.0.0',
        'engines': engine_health,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'uptime': 'operational'
    })

@app.route('/api/info', methods=['GET'])
def gateway_info():
    """Gateway information"""
    return jsonify({
        'service': 'Multi-Language Risk Simulation API Gateway',
        'version': '1.0.0',
        'description': 'Unified interface to Node.js and Python risk simulation engines',
        
        'engines': {
            'node': {
                'url': NODE_SERVER_URL,
                'status': engine_health['node']['status'],
                'specialties': ['High-performance Monte Carlo', 'Real-time simulations', 'Web integration'],
                'preferred_methods': ['monte_carlo', 'historical_simulation', 'bootstrap', 'variance_covariance']
            },
            'python': {
                'url': PYTHON_SERVER_URL,
                'status': engine_health['python']['status'],
                'specialties': ['Scientific computing', 'Advanced analytics', 'Research methods'],
                'preferred_methods': ['advanced_monte_carlo', 'advanced_bootstrap', 'gbm', 'multi_asset_gbm', 'path_dependent_gbm']
            }
        },
        
        'features': {
            'automatic_engine_selection': True,
            'load_balancing': True,
            'health_monitoring': True,
            'failover_support': True,
            'unified_response_format': True
        },
        
        'endpoints': {
            '/simulate': 'Main simulation endpoint with automatic engine selection',
            '/compare': 'Compare multiple simulation methods across engines',
            '/benchmark': 'Performance benchmarking across engines',
            '/engines/{engine}/simulate': 'Force specific engine usage'
        }
    })

@app.route('/api/engines', methods=['GET'])
def list_engines():
    """List available engines and their status"""
    return jsonify({
        'engines': {
            'node': {
                'status': engine_health['node']['status'],
                'url': NODE_SERVER_URL,
                'response_time': engine_health['node']['response_time'],
                'last_check': engine_health['node']['last_check'],
                'capabilities': ['monte_carlo', 'historical_simulation', 'bootstrap', 'variance_covariance', 'gbm']
            },
            'python': {
                'status': engine_health['python']['status'],
                'url': PYTHON_SERVER_URL,
                'response_time': engine_health['python']['response_time'],
                'last_check': engine_health['python']['last_check'],
                'capabilities': ['monte_carlo', 'advanced_monte_carlo', 'gbm', 'multi_asset_gbm', 'path_dependent_gbm', 'stress_test_varcov']
            }
        },
        'selection_algorithm': {
            'factors': ['user_preference', 'method_compatibility', 'engine_health', 'performance'],
            'fallback_enabled': True
        }
    })

@app.route('/simulate', methods=['POST'])
def simulate():
    """Main simulation endpoint with automatic engine selection"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        method = data.get('method')
        requested_engine = data.get('engine')  # Optional engine preference
        
        if not method:
            return jsonify({'error': 'Missing required parameter: method'}), 400
        
        # Select the best engine
        selected_engine = select_engine(requested_engine, method)
        print(f"🎯 Selected {selected_engine} engine for {method} simulation")
        
        # Ensure engine is specified in the request
        data['engine'] = selected_engine
        
        # Forward to selected engine
        result = forward_request(selected_engine, '/simulate', data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Gateway error: {str(e)}',
            'gateway_info': {
                'error_in_gateway': True,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        }), 500

@app.route('/engines/<engine>/simulate', methods=['POST'])
def simulate_with_engine(engine: str):
    """Force simulation with specific engine"""
    try:
        if engine not in ['node', 'python']:
            return jsonify({
                'error': f'Unknown engine: {engine}',
                'available_engines': ['node', 'python']
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        # Force the specified engine
        data['engine'] = engine
        
        print(f"🎯 Forcing {engine} engine for simulation")
        
        # Forward to specified engine
        result = forward_request(engine, '/simulate', data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Gateway error: {str(e)}',
            'forced_engine': engine
        }), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Compare methods across engines"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        methods = data.get('methods', [])
        if not methods:
            return jsonify({'error': 'No methods specified for comparison'}), 400
        
        # Determine which engines to use
        engines_to_use = set()
        for method in methods:
            best_engine = select_engine(None, method)
            engines_to_use.add(best_engine)
        
        print(f"🔍 Running comparison across engines: {list(engines_to_use)}")
        
        # If only one engine needed, use its native comparison
        if len(engines_to_use) == 1:
            engine = list(engines_to_use)[0]
            data['engine'] = engine
            result = forward_request(engine, '/compare', data)
            return jsonify(result)
        
        # Multi-engine comparison
        all_results = {}
        
        for engine in engines_to_use:
            print(f"  Running comparison on {engine} engine...")
            engine_data = data.copy()
            engine_data['engine'] = engine
            
            # Filter methods compatible with this engine
            engine_methods = []
            for method in methods:
                if select_engine(None, method) == engine:
                    engine_methods.append(method)
            
            if engine_methods:
                engine_data['methods'] = engine_methods
                engine_result = forward_request(engine, '/compare', engine_data)
                all_results[engine] = engine_result
        
        # Combine results
        combined_result = {
            'multi_engine_comparison': all_results,
            'gateway_info': {
                'engines_used': list(engines_to_use),
                'comparison_type': 'multi_engine',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        }
        
        return jsonify(combined_result)
        
    except Exception as e:
        return jsonify({
            'error': f'Gateway comparison error: {str(e)}',
            'gateway_info': {
                'error_in_gateway': True
            }
        }), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Benchmark performance across engines"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        method = data.get('method', 'monte_carlo')
        
        # Run benchmark on both engines if healthy
        results = {}
        
        for engine in ['node', 'python']:
            if engine_health[engine]['status'] == 'healthy':
                print(f"🏃 Benchmarking {method} on {engine} engine...")
                engine_data = data.copy()
                engine_data['engine'] = engine
                
                benchmark_result = forward_request(engine, '/benchmark', engine_data)
                results[engine] = benchmark_result
        
        # Cross-engine performance analysis
        if len(results) > 1:
            node_throughput = 0
            python_throughput = 0
            
            if 'node' in results and 'benchmark' in results['node']:
                node_results = results['node']['benchmark'].get('results', [])
                if node_results:
                    node_throughput = max([r.get('throughput', 0) for r in node_results])
            
            if 'python' in results and 'benchmark' in results['python']:
                python_results = results['python']['benchmark'].get('results', [])
                if python_results:
                    python_throughput = max([r.get('throughput', 0) for r in python_results])
            
            performance_comparison = {
                'node_max_throughput': node_throughput,
                'python_max_throughput': python_throughput,
                'faster_engine': 'node' if node_throughput > python_throughput else 'python',
                'performance_ratio': max(node_throughput, python_throughput) / min(node_throughput, python_throughput) if min(node_throughput, python_throughput) > 0 else 1.0
            }
        else:
            performance_comparison = {'note': 'Only one engine available for benchmarking'}
        
        combined_result = {
            'cross_engine_benchmark': results,
            'performance_analysis': performance_comparison,
            'gateway_info': {
                'engines_tested': list(results.keys()),
                'method': method,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        }
        
        return jsonify(combined_result)
        
    except Exception as e:
        return jsonify({
            'error': f'Gateway benchmark error: {str(e)}',
            'gateway_info': {
                'error_in_gateway': True
            }
        }), 500

# Proxy endpoints for engine-specific documentation
@app.route('/api/docs', methods=['GET'])
def gateway_docs():
    """Gateway API documentation"""
    return jsonify({
        'title': 'Multi-Language Risk Simulation API Gateway',
        'description': 'Unified access to Node.js and Python risk simulation engines',
        'version': '1.0.0',
        
        'gateway_endpoints': {
            '/health': 'Gateway health and engine status',
            '/api/info': 'Gateway and engine information',
            '/api/engines': 'List available engines and capabilities',
            '/simulate': 'Automatic engine selection for simulation',
            '/engines/{engine}/simulate': 'Force specific engine usage',
            '/compare': 'Cross-engine method comparison',
            '/benchmark': 'Cross-engine performance benchmarking'
        },
        
        'engine_selection': {
            'automatic': 'Based on method compatibility, health, and performance',
            'manual': 'Use /engines/{engine}/simulate to force specific engine',
            'fallback': 'Automatic fallback if preferred engine is unhealthy'
        },
        
        'supported_engines': {
            'node': {
                'documentation': f'{NODE_SERVER_URL}/api/docs',
                'examples': f'{NODE_SERVER_URL}/api/examples',
                'status': engine_health['node']['status']
            },
            'python': {
                'documentation': f'{PYTHON_SERVER_URL}/api/docs',
                'examples': f'{PYTHON_SERVER_URL}/api/examples',
                'status': engine_health['python']['status']
            }
        },
        
        'example_request': {
            'method': 'POST',
            'url': '/simulate',
            'body': {
                'method': 'monte_carlo',
                'portfolio': [100000, 200000, 150000, 50000],
                'params': {
                    'iterations': 1000000,
                    'confidence': 0.95,
                    'horizon': 1.0
                }
            },
            'note': 'Engine will be automatically selected based on method and availability'
        }
    })

if __name__ == '__main__':
    print(f"🌐 Multi-Language Risk Simulation API Gateway starting on port {GATEWAY_PORT}")
    print(f"📊 Node.js Engine: {NODE_SERVER_URL}")
    print(f"🐍 Python Engine: {PYTHON_SERVER_URL}")
    print(f"🎯 Automatic engine selection enabled")
    print(f"📖 Gateway Documentation: http://localhost:{GATEWAY_PORT}/api/docs")
    print(f"🏥 Gateway Health: http://localhost:{GATEWAY_PORT}/health")
    print(f"🚀 Ready to route risk simulations!")
    
    app.run(host='0.0.0.0', port=GATEWAY_PORT, debug=False)
