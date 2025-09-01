"""
Startup script for the Multi-Language Risk Simulation Engine

This script starts all components of the risk simulation system:
- Node.js backend server (port 3001)
- Python backend server (port 3002) 
- API Gateway (port 3000)

Usage:
    python start_all.py [--dev] [--node-only] [--python-only] [--gateway-only]
"""

import subprocess
import time
import sys
import threading
import os
import signal
import argparse
from typing import List, Dict

class ProcessManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True
        
    def start_process(self, command: List[str], name: str, cwd: str = None) -> subprocess.Popen:
        """Start a subprocess and track it"""
        try:
            print(f"🚀 Starting {name}...")
            
            # For Windows compatibility
            if sys.platform == "win32":
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(command, cwd=cwd, preexec_fn=os.setsid)
            
            self.processes.append(process)
            print(f"✅ {name} started (PID: {process.pid})")
            return process
            
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return None

    def stop_all_processes(self):
        """Stop all tracked processes"""
        print("\n🛑 Stopping all processes...")
        self.running = False
        
        for process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    if sys.platform == "win32":
                        # Windows
                        process.terminate()
                    else:
                        # Unix-like
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    process.wait(timeout=5)
                    print(f"✅ Process {process.pid} stopped gracefully")
                    
            except subprocess.TimeoutExpired:
                # Force kill if didn't stop gracefully
                if sys.platform == "win32":
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                print(f"🔨 Process {process.pid} force killed")
            except Exception as e:
                print(f"⚠️  Error stopping process {process.pid}: {e}")
        
        self.processes.clear()
        print("🏁 All processes stopped")

def check_prerequisites():
    """Check if required tools are installed"""
    requirements = {
        'node': ['node', '--version'],
        'npm': ['npm', '--version'], 
        'python': ['python', '--version']
    }
    
    missing = []
    
    for tool, command in requirements.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ {tool}: {version}")
            else:
                missing.append(tool)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing required tools: {', '.join(missing)}")
        print("Please install the missing tools and try again.")
        return False
    
    return True

def install_dependencies(manager: ProcessManager):
    """Install dependencies for all components"""
    print("📦 Installing dependencies...")
    
    # Install root dependencies
    print("  Installing root dependencies...")
    subprocess.run(['npm', 'install'], check=True)
    
    # Install Node.js backend dependencies
    print("  Installing Node.js backend dependencies...")
    subprocess.run(['npm', 'install'], cwd='backend/node', check=True)
    
    # Install Python backend dependencies
    print("  Installing Python backend dependencies...")
    subprocess.run([
        'pip', 'install', '-r', 'requirements.txt'
    ], cwd='backend/python', check=True)
    
    print("✅ All dependencies installed")

def wait_for_health_check(url: str, service_name: str, timeout: int = 60):
    """Wait for a service to become healthy"""
    import requests
    
    print(f"🏥 Waiting for {service_name} to become healthy...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    print(f"⚠️  {service_name} health check timed out")
    return False

def main():
    parser = argparse.ArgumentParser(description='Start Multi-Language Risk Simulation Engine')
    parser.add_argument('--dev', action='store_true', help='Start in development mode')
    parser.add_argument('--node-only', action='store_true', help='Start only Node.js backend')
    parser.add_argument('--python-only', action='store_true', help='Start only Python backend')
    parser.add_argument('--gateway-only', action='store_true', help='Start only API gateway')
    parser.add_argument('--no-install', action='store_true', help='Skip dependency installation')
    parser.add_argument('--no-health-check', action='store_true', help='Skip health checks')
    
    args = parser.parse_args()
    
    print("🌟 Multi-Language Risk Simulation Engine Startup")
    print("=" * 55)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Create process manager
    manager = ProcessManager()
    
    def signal_handler(sig, frame):
        manager.stop_all_processes()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Install dependencies unless skipped
        if not args.no_install:
            install_dependencies(manager)
        
        # Determine what to start
        start_node = not (args.python_only or args.gateway_only)
        start_python = not (args.node_only or args.gateway_only)
        start_gateway = not (args.node_only or args.python_only)
        
        processes = {}
        
        # Start Node.js backend
        if start_node:
            node_command = ['npm', 'run', 'dev' if args.dev else 'start']
            processes['node'] = manager.start_process(
                node_command, 
                "Node.js Backend (port 3001)",
                cwd='backend/node'
            )
            time.sleep(3)  # Give it time to start
        
        # Start Python backend
        if start_python:
            python_env = os.environ.copy()
            if args.dev:
                python_env['FLASK_DEBUG'] = 'true'
            
            processes['python'] = manager.start_process(
                ['python', 'app.py'],
                "Python Backend (port 3002)",
                cwd='backend/python'
            )
            time.sleep(3)  # Give it time to start
        
        # Start API Gateway
        if start_gateway:
            processes['gateway'] = manager.start_process(
                ['python', 'gateway.py'],
                "API Gateway (port 3000)"
            )
            time.sleep(2)  # Give it time to start
        
        # Health checks unless skipped
        if not args.no_health_check:
            print("\\n🏥 Performing health checks...")
            
            try:
                import requests
                
                if start_node:
                    wait_for_health_check('http://localhost:3001/health', 'Node.js Backend')
                
                if start_python:
                    wait_for_health_check('http://localhost:3002/health', 'Python Backend')
                
                if start_gateway:
                    wait_for_health_check('http://localhost:3000/health', 'API Gateway')
                    
            except ImportError:
                print("⚠️  'requests' module not available, skipping health checks")
        
        # Display startup summary
        print("\\n" + "=" * 55)
        print("🎉 Multi-Language Risk Simulation Engine is running!")
        print("=" * 55)
        
        if start_node:
            print("📊 Node.js Backend:    http://localhost:3001")
            print("    Documentation:    http://localhost:3001/api/docs")
        
        if start_python:
            print("🐍 Python Backend:     http://localhost:3002") 
            print("    Documentation:    http://localhost:3002/api/docs")
        
        if start_gateway:
            print("🌐 API Gateway:        http://localhost:3000")
            print("    Documentation:    http://localhost:3000/api/docs")
            print("    Health Status:    http://localhost:3000/health")
        
        print("\\n💡 Tips:")
        if start_gateway:
            print("   • Use the API Gateway (port 3000) for automatic engine selection")
            print("   • Send requests to /simulate for automatic routing")
        print("   • Press Ctrl+C to stop all services")
        print("   • Check logs for any issues")
        
        # Example API calls
        if start_gateway:
            print("\\n🔧 Example API Call:")
            print("""
curl -X POST http://localhost:3000/simulate \\
  -H "Content-Type: application/json" \\
  -d '{
    "method": "monte_carlo",
    "portfolio": [100000, 200000, 150000, 50000],
    "params": {
      "iterations": 100000,
      "confidence": 0.95,
      "horizon": 1.0
    }
  }'
            """)
        
        # Keep running until interrupted
        print("\\n🔄 Services running... Press Ctrl+C to stop")
        while manager.running:
            time.sleep(1)
            
            # Check if any process died
            for name, process in list(processes.items()):
                if process and process.poll() is not None:
                    print(f"⚠️  {name} process stopped unexpectedly (exit code: {process.returncode})")
                    processes[name] = None
            
    except KeyboardInterrupt:
        print("\\n⏹️  Received interrupt signal")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
    finally:
        manager.stop_all_processes()

if __name__ == '__main__':
    main()
