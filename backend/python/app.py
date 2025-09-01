"""
Risk Simulation Engine - Python Backend Server

Flask server providing REST API for multiple risk simulation methods:
- Monte Carlo Simulation
- Historical Simulation  
- Bootstrap Simulation
- Variance-Covariance Method
- Geometric Brownian Motion

This server demonstrates high-performance financial risk modeling
using Python with comprehensive simulation strategies.
"""

from flask import Flask
from flask_cors import CORS
import os
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure Flask
app.config['JSON_SORT_KEYS'] = False

# Import and register routes
from routes.routes import routes_bp
app.register_blueprint(routes_bp)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return {
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'GET /api/info', 
            'GET /api/docs',
            'GET /api/examples',
            'POST /simulate',
            'POST /compare',
            'POST /benchmark'
        ],
        'engine': 'python'
    }, 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return {
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'engine': 'python'
    }, 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3002))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🐍 Risk Simulation Engine (Python) starting on port {port}")
    print(f"📊 API Documentation: http://localhost:{port}/api/docs")
    print(f"🏥 Health Check: http://localhost:{port}/health")
    print(f"📝 Examples: http://localhost:{port}/api/examples")
    print(f"🎯 Ready to process risk simulations!")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
