import { motion } from "framer-motion";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Select, SelectItem } from "./components/ui/select";

// Utility function to normalize simulation results to a consistent format
const normalizeSimulationResult = (data) => {
    if (!data || data.error) return data;

    // Create a normalized result object
    const normalized = {
        ...data,
        // Extract VaR from various possible locations
        VaR_95: data.VaR_95 ||
            (data.risk_metrics && data.risk_metrics.VaR_95) ||
            (data.var_95) ||
            null,

        // Extract CVaR from various possible locations  
        CVaR_95: data.CVaR_95 ||
            (data.risk_metrics && data.risk_metrics.CVaR_95) ||
            (data.cvar_95) ||
            (data.risk_metrics && data.risk_metrics.expected_shortfall) ||
            null,

        // Extract expected return from various locations
        expected_return: data.expected_return ||
            (data.portfolio_stats && data.portfolio_stats.expected_return) ||
            (data.portfolio_analytics && data.portfolio_analytics.portfolio_return) ||
            (data.expected_terminal_value) ||
            null,

        // Extract volatility from various locations
        volatility: data.volatility ||
            (data.portfolio_stats && data.portfolio_stats.volatility) ||
            (data.portfolio_analytics && data.portfolio_analytics.portfolio_volatility) ||
            null,

        // Extract execution time from various locations
        execution_time: data.execution_time ||
            (data.performance && data.performance.simulation_time) ||
            (data.performance && data.performance.calculation_time) ||
            (data.server_info && data.server_info.total_processing_time) ||
            null,

        // Extract method name from various locations
        method: data.method ||
            (data.simulation_config && data.simulation_config.method) ||
            (data.server_info && data.server_info.method) ||
            'unknown',

        // Extract engine from various locations
        engine: data.engine ||
            (data.simulation_config && data.simulation_config.engine) ||
            (data.server_info && data.server_info.engine) ||
            'unknown',

        // Extract distribution data
        distribution: data.distribution ||
            (data.simulation_results && data.simulation_results.distribution) ||
            null,

        // Add additional metrics for display
        sharpe_ratio: data.sharpe_ratio ||
            (data.portfolio_stats && data.portfolio_stats.sharpe_ratio) ||
            (data.portfolio_analytics && data.portfolio_analytics.sharpe_ratio) ||
            null,

        max_loss: data.maximum_loss ||
            (data.risk_metrics && data.risk_metrics.maximum_loss) ||
            null,

        max_gain: data.maximum_gain ||
            (data.risk_metrics && data.risk_metrics.maximum_gain) ||
            null,

        probability_of_loss: data.probability_of_loss ||
            (data.risk_metrics && data.risk_metrics.probability_of_loss) ||
            null
    };

    return normalized;
};

export default function App() {
    const [engine, setEngine] = useState("python");
    const [method, setMethod] = useState("monte_carlo");
    const [portfolio, setPortfolio] = useState("100000,200000,150000");
    const [iterations, setIterations] = useState(100000);
    const [confidence, setConfidence] = useState(0.95);
    const [horizon, setHorizon] = useState(1.0);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Comparison mode
    const [comparisonMode, setComparisonMode] = useState(false);
    const [selectedMethods, setSelectedMethods] = useState(['monte_carlo', 'historical_simulation', 'variance_covariance']);
    const [comparisonResults, setComparisonResults] = useState(null);

    // Advanced parameters for different methods
    const [advancedMode, setAdvancedMode] = useState(false);
    const [expectedReturns, setExpectedReturns] = useState("0.08,0.06,0.10");
    const [volatilities, setVolatilities] = useState("0.20,0.15,0.25");
    const [correlationMatrix, setCorrelationMatrix] = useState("1.0,0.3,0.2|0.3,1.0,0.4|0.2,0.4,1.0");

    // Method complexity and performance info
    const getMethodInfo = (method) => {
        const methodInfo = {
            'monte_carlo': {
                complexity: 'Medium',
                recommended_iterations: 100000,
                description: 'Basic Monte Carlo with Geometric Brownian Motion',
                gpu_optimized: true
            },
            'advanced_monte_carlo': {
                complexity: 'High',
                recommended_iterations: 500000,
                description: 'Advanced Monte Carlo with correlation modeling',
                gpu_optimized: true
            },
            'gbm': {
                complexity: 'Medium',
                recommended_iterations: 100000,
                description: 'Geometric Brownian Motion with time steps',
                gpu_optimized: true
            },
            'multi_asset_gbm': {
                complexity: 'High',
                recommended_iterations: 250000,
                description: 'Multi-asset GBM with jump diffusion',
                gpu_optimized: true
            },
            'path_dependent_gbm': {
                complexity: 'High',
                recommended_iterations: 100000,
                description: 'Path-dependent GBM with barriers',
                gpu_optimized: false
            },
            'historical_simulation': {
                complexity: 'Low',
                recommended_iterations: 100000,
                description: 'Non-parametric historical resampling',
                gpu_optimized: false
            },
            'bootstrap_historical': {
                complexity: 'Medium',
                recommended_iterations: 50000,
                description: 'Bootstrap resampling of historical data',
                gpu_optimized: false
            },
            'bootstrap': {
                complexity: 'Medium',
                recommended_iterations: 50000,
                description: 'Basic bootstrap simulation',
                gpu_optimized: false
            },
            'advanced_bootstrap': {
                complexity: 'High',
                recommended_iterations: 500000,
                description: 'Stationary bootstrap with blocks',
                gpu_optimized: false
            },
            'variance_covariance': {
                complexity: 'Low',
                recommended_iterations: null,
                description: 'Analytical parametric method',
                gpu_optimized: false
            },
            'stress_test_varcov': {
                complexity: 'Low',
                recommended_iterations: null,
                description: 'Stress testing with VarCov',
                gpu_optimized: false
            }
        };
        return methodInfo[method] || methodInfo['monte_carlo'];
    };

    // Auto-adjust iterations based on method
    const handleMethodChange = (newMethod) => {
        setMethod(newMethod);
        const methodInfo = getMethodInfo(newMethod);
        if (methodInfo.recommended_iterations) {
            setIterations(methodInfo.recommended_iterations);
        }
    };

    // Method-specific parameter builder
    const buildMethodParams = () => {
        const portfolioArray = portfolio.split(",").map((v) => parseFloat(v.trim()));
        const n_assets = portfolioArray.length;

        let params = {
            iterations: parseInt(iterations),
            confidence: parseFloat(confidence),
            horizon: parseFloat(horizon)
        };

        // Add advanced parameters if enabled
        if (advancedMode) {
            const returns = expectedReturns.split(",").map(v => parseFloat(v.trim()));
            const vols = volatilities.split(",").map(v => parseFloat(v.trim()));

            // Parse correlation matrix
            const corrRows = correlationMatrix.split("|");
            const corrMatrix = corrRows.map(row =>
                row.split(",").map(v => parseFloat(v.trim()))
            );

            params.expected_returns = returns;
            params.volatilities = vols;
            params.correlation_matrix = corrMatrix;
        }

        // Method-specific parameters
        switch (method) {
            case 'gbm':
                params.time_steps = 252;
                break;
            case 'multi_asset_gbm':
                params.jump_intensity = 0.1;
                params.jump_mean = -0.05;
                params.jump_std = 0.15;
                params.correlations = 0.3;
                break;
            case 'path_dependent_gbm':
                params.barrier_level = 0.8;
                params.barrier_type = "down_and_out";
                params.lookback_monitoring = true;
                break;
            case 'bootstrap_historical':
                params.bootstrap_samples = Math.min(parseInt(iterations), 50000);
                params.block_size = 5;
                params.lookback_days = 750;
                break;
            case 'advanced_bootstrap':
                params.block_size = 10;
                params.bootstrap_method = "stationary";
                params.sample_size = 1000;
                break;
            case 'historical_simulation':
                params.lookback_days = 1000;
                params.data_source = "random_generated";
                break;
            case 'stress_test_varcov':
                params.stress_scenarios = [
                    { "market_shock": -0.20, "volatility_increase": 1.5 },
                    { "correlation_increase": 0.3, "liquidity_stress": 0.1 }
                ];
                break;
        }

        return params;
    };

    const runSimulation = async () => {
        setLoading(true);
        setResult(null);

        const body = {
            engine,
            method,
            portfolio: portfolio.split(",").map((v) => parseFloat(v.trim())),
            params: buildMethodParams()
        };

        try {
            const res = await fetch(`http://localhost:3002/simulate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await res.json();

            // Normalize the result to handle different output formats
            const normalizedResult = normalizeSimulationResult(data);
            setResult(normalizedResult);
        } catch (err) {
            console.error("Error:", err);
            setResult({ error: "Failed to connect to simulation engine. Please ensure the backend server is running on port 3002." });
        }
        setLoading(false);
    };

    const runComparison = async () => {
        setLoading(true);
        setComparisonResults(null);
        setResult(null);

        const body = {
            portfolio: portfolio.split(",").map((v) => parseFloat(v.trim())),
            methods: selectedMethods,
            params: buildMethodParams()
        };

        try {
            const res = await fetch(`http://localhost:3002/compare`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await res.json();

            // Normalize each method's results
            if (data.results) {
                const normalizedResults = {};
                Object.keys(data.results).forEach(methodName => {
                    normalizedResults[methodName] = normalizeSimulationResult(data.results[methodName]);
                });
                setComparisonResults({
                    ...data,
                    results: normalizedResults
                });
            } else {
                setComparisonResults(data);
            }
        } catch (err) {
            console.error("Comparison Error:", err);
            setComparisonResults({ error: "Failed to run comparison. Please ensure the backend server is running on port 3002." });
        }
        setLoading(false);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center p-6">
            <motion.div
                className="w-full max-w-4xl"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
            >
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-gray-800 mb-2">
                        📊 Risk Simulation Engine
                    </h1>
                    <p className="text-lg text-gray-600">
                        Advanced Financial Risk Modeling with Multiple Simulation Methods
                    </p>
                </div>

                {/* Main Form Card */}
                <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm rounded-2xl mb-6">
                    <CardContent className="p-8 space-y-6">
                        {/* Mode Selection */}
                        <div className="flex items-center justify-center space-x-4 p-4 bg-gray-50 rounded-lg border">
                            <div className="flex items-center space-x-2">
                                <input
                                    type="radio"
                                    id="single-mode"
                                    name="simulation-mode"
                                    checked={!comparisonMode}
                                    onChange={() => setComparisonMode(false)}
                                    className="text-blue-600"
                                />
                                <label htmlFor="single-mode" className="text-sm font-medium text-gray-700">
                                    🎯 Single Method
                                </label>
                            </div>
                            <div className="flex items-center space-x-2">
                                <input
                                    type="radio"
                                    id="comparison-mode"
                                    name="simulation-mode"
                                    checked={comparisonMode}
                                    onChange={() => setComparisonMode(true)}
                                    className="text-blue-600"
                                />
                                <label htmlFor="comparison-mode" className="text-sm font-medium text-gray-700">
                                    ⚔️ Method Comparison
                                </label>
                            </div>
                        </div>

                        {!comparisonMode ? (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Engine Select */}
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-gray-700">Engine</label>
                                    <Select value={engine} onValueChange={setEngine}>
                                        <SelectItem value="python">🐍 Python Engine</SelectItem>
                                        <SelectItem value="node">🟢 Node.js Engine</SelectItem>
                                    </Select>
                                </div>

                                {/* Method Select */}
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-gray-700">Simulation Method</label>
                                    <Select value={method} onValueChange={handleMethodChange}>
                                        {/* Monte Carlo Methods */}
                                        <SelectItem value="monte_carlo">🎲 Basic Monte Carlo</SelectItem>
                                        <SelectItem value="advanced_monte_carlo">🎯 Advanced Monte Carlo</SelectItem>

                                        {/* Geometric Brownian Motion */}
                                        <SelectItem value="gbm">🌊 Basic GBM</SelectItem>
                                        <SelectItem value="multi_asset_gbm">🌊 Multi-Asset GBM</SelectItem>
                                        <SelectItem value="path_dependent_gbm">🛤️ Path-Dependent GBM</SelectItem>

                                        {/* Historical Methods */}
                                        <SelectItem value="historical_simulation">📈 Historical Simulation</SelectItem>
                                        <SelectItem value="bootstrap_historical">📈 Bootstrap Historical</SelectItem>

                                        {/* Bootstrap Methods */}
                                        <SelectItem value="bootstrap">🔄 Basic Bootstrap</SelectItem>
                                        <SelectItem value="advanced_bootstrap">🔄 Advanced Bootstrap</SelectItem>

                                        {/* Variance-Covariance */}
                                        <SelectItem value="variance_covariance">📊 Variance-Covariance</SelectItem>
                                        <SelectItem value="stress_test_varcov">⚠️ Stress Test VarCov</SelectItem>
                                    </Select>

                                    {/* Method Information */}
                                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                                        <div className="flex items-center justify-between">
                                            <span className="text-gray-600">
                                                <span className="font-semibold">Complexity:</span> {getMethodInfo(method).complexity}
                                            </span>
                                            {getMethodInfo(method).gpu_optimized && (
                                                <span className="text-green-600 font-semibold">🚀 GPU Optimized</span>
                                            )}
                                        </div>
                                        <p className="text-gray-700 mt-1">{getMethodInfo(method).description}</p>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <h3 className="text-lg font-semibold text-gray-800">Select Methods to Compare</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                    {[
                                        { value: 'monte_carlo', label: '🎲 Basic Monte Carlo', fast: true },
                                        { value: 'advanced_monte_carlo', label: '🎯 Advanced Monte Carlo', fast: false },
                                        { value: 'historical_simulation', label: '📈 Historical Simulation', fast: true },
                                        { value: 'variance_covariance', label: '📊 Variance-Covariance', fast: true },
                                        { value: 'bootstrap', label: '🔄 Bootstrap', fast: true },
                                        { value: 'gbm', label: '🌊 Basic GBM', fast: true }
                                    ].map((methodOption) => (
                                        <div key={methodOption.value} className="flex items-center space-x-2 p-2 border rounded hover:bg-gray-50">
                                            <input
                                                type="checkbox"
                                                id={methodOption.value}
                                                checked={selectedMethods.includes(methodOption.value)}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        setSelectedMethods([...selectedMethods, methodOption.value]);
                                                    } else {
                                                        setSelectedMethods(selectedMethods.filter(m => m !== methodOption.value));
                                                    }
                                                }}
                                                className="text-blue-600"
                                            />
                                            <label htmlFor={methodOption.value} className="text-sm text-gray-700 cursor-pointer flex-1">
                                                {methodOption.label}
                                                {methodOption.fast && <span className="text-green-600 text-xs ml-1">⚡</span>}
                                            </label>
                                        </div>
                                    ))}
                                </div>
                                <p className="text-xs text-gray-500">
                                    ⚡ = Fast methods recommended for comparison. Select 2-6 methods for best results.
                                </p>
                            </div>
                        )}

                        {/* Portfolio Input */}
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-700">
                                Portfolio Values (comma-separated)
                            </label>
                            <Input
                                value={portfolio}
                                onChange={(e) => setPortfolio(e.target.value)}
                                placeholder="100000,200000,150000"
                                className="text-sm"
                            />
                            <p className="text-xs text-gray-500">
                                Enter portfolio values separated by commas (e.g., 100000,200000,150000)
                            </p>
                        </div>

                        {/* Parameters */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700">Iterations</label>
                                <Input
                                    type="number"
                                    value={iterations}
                                    onChange={(e) => setIterations(e.target.value)}
                                    min="1000"
                                    max="10000000"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700">Confidence Level</label>
                                <Input
                                    type="number"
                                    step="0.01"
                                    min="0.01"
                                    max="0.99"
                                    value={confidence}
                                    onChange={(e) => setConfidence(e.target.value)}
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700">Time Horizon (Years)</label>
                                <Input
                                    type="number"
                                    step="0.1"
                                    min="0.1"
                                    max="10"
                                    value={horizon}
                                    onChange={(e) => setHorizon(e.target.value)}
                                />
                            </div>
                        </div>

                        {/* Advanced Parameters Toggle */}
                        <div className="flex items-center space-x-2">
                            <input
                                type="checkbox"
                                id="advanced-mode"
                                checked={advancedMode}
                                onChange={(e) => setAdvancedMode(e.target.checked)}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <label htmlFor="advanced-mode" className="text-sm font-medium text-gray-700">
                                🔧 Advanced Parameters
                            </label>
                        </div>

                        {/* Advanced Parameters Section */}
                        {advancedMode && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                exit={{ opacity: 0, height: 0 }}
                                className="space-y-4 p-4 bg-gray-50 rounded-lg border"
                            >
                                <h3 className="text-sm font-semibold text-gray-800">Advanced Simulation Parameters</h3>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium text-gray-700">Expected Returns</label>
                                        <Input
                                            value={expectedReturns}
                                            onChange={(e) => setExpectedReturns(e.target.value)}
                                            placeholder="0.08,0.06,0.10"
                                            className="text-sm"
                                        />
                                        <p className="text-xs text-gray-500">Comma-separated annual returns (e.g., 0.08,0.06,0.10)</p>
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-sm font-medium text-gray-700">Volatilities</label>
                                        <Input
                                            value={volatilities}
                                            onChange={(e) => setVolatilities(e.target.value)}
                                            placeholder="0.20,0.15,0.25"
                                            className="text-sm"
                                        />
                                        <p className="text-xs text-gray-500">Comma-separated volatilities (e.g., 0.20,0.15,0.25)</p>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-gray-700">Correlation Matrix</label>
                                    <Input
                                        value={correlationMatrix}
                                        onChange={(e) => setCorrelationMatrix(e.target.value)}
                                        placeholder="1.0,0.3,0.2|0.3,1.0,0.4|0.2,0.4,1.0"
                                        className="text-sm"
                                    />
                                    <p className="text-xs text-gray-500">
                                        Correlation matrix rows separated by | (e.g., 1.0,0.3,0.2|0.3,1.0,0.4|0.2,0.4,1.0)
                                    </p>
                                </div>

                                <div className="bg-blue-50 p-3 rounded border border-blue-200">
                                    <p className="text-xs text-blue-700">
                                        <span className="font-semibold">💡 Tip:</span> Advanced parameters override defaults.
                                        Different simulation methods will automatically use appropriate parameters even if not all are specified.
                                    </p>
                                </div>
                            </motion.div>
                        )}

                        {/* Run Button */}
                        <motion.div
                            className="pt-4"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            <Button
                                onClick={comparisonMode ? runComparison : runSimulation}
                                disabled={loading || (comparisonMode && selectedMethods.length < 2)}
                                className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl shadow-lg disabled:opacity-50"
                            >
                                {loading ? (
                                    <div className="flex items-center space-x-2">
                                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                                        <span>{comparisonMode ? 'Comparing Methods...' : 'Running Simulation...'}</span>
                                    </div>
                                ) : (
                                    comparisonMode ?
                                        `⚔️ Compare ${selectedMethods.length} Methods` :
                                        "🚀 Run Risk Simulation"
                                )}
                            </Button>
                            {comparisonMode && selectedMethods.length < 2 && (
                                <p className="text-sm text-red-600 mt-2 text-center">
                                    Please select at least 2 methods to compare
                                </p>
                            )}
                        </motion.div>
                    </CardContent>
                </Card>

                {/* Results Section */}
                {result && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm rounded-2xl">
                            <CardContent className="p-8">
                                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                                    📊 Simulation Results
                                </h2>

                                {result.error ? (
                                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                        <p className="text-red-700 font-medium">⚠️ Error</p>
                                        <p className="text-red-600 text-sm mt-1">{result.error}</p>
                                    </div>
                                ) : (
                                    <div className="space-y-6">
                                        {/* Metadata */}
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                                            <div>
                                                <span className="text-sm font-medium text-gray-600">Engine:</span>
                                                <span className="ml-2 text-sm text-gray-800 font-semibold">
                                                    {result.engine === 'python' ? '🐍 Python' : '🟢 Node.js'}
                                                </span>
                                            </div>
                                            <div>
                                                <span className="text-sm font-medium text-gray-600">Method:</span>
                                                <span className="ml-2 text-sm text-gray-800 font-semibold">
                                                    {result.method}
                                                </span>
                                            </div>
                                        </div>

                                        {/* Risk Metrics */}
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                            {result.VaR_95 && (
                                                <div className="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-lg border border-red-200">
                                                    <p className="text-sm font-medium text-red-700">VaR (95%)</p>
                                                    <p className="text-xl font-bold text-red-800">
                                                        ${Math.abs(result.VaR_95).toLocaleString()}
                                                    </p>
                                                </div>
                                            )}

                                            {result.CVaR_95 && (
                                                <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg border border-orange-200">
                                                    <p className="text-sm font-medium text-orange-700">CVaR (95%)</p>
                                                    <p className="text-xl font-bold text-orange-800">
                                                        ${Math.abs(result.CVaR_95).toLocaleString()}
                                                    </p>
                                                </div>
                                            )}

                                            {result.expected_return && (
                                                <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                                                    <p className="text-sm font-medium text-green-700">Expected Return</p>
                                                    <p className="text-xl font-bold text-green-800">
                                                        {typeof result.expected_return === 'number' ?
                                                            (result.expected_return > 1000 ?
                                                                `$${result.expected_return.toLocaleString()}` :
                                                                `${(result.expected_return * 100).toFixed(2)}%`
                                                            ) :
                                                            result.expected_return
                                                        }
                                                    </p>
                                                </div>
                                            )}

                                            {result.volatility && (
                                                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
                                                    <p className="text-sm font-medium text-blue-700">Volatility</p>
                                                    <p className="text-xl font-bold text-blue-800">
                                                        {(result.volatility * 100).toFixed(2)}%
                                                    </p>
                                                </div>
                                            )}

                                            {result.sharpe_ratio && (
                                                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
                                                    <p className="text-sm font-medium text-purple-700">Sharpe Ratio</p>
                                                    <p className="text-xl font-bold text-purple-800">
                                                        {result.sharpe_ratio.toFixed(3)}
                                                    </p>
                                                </div>
                                            )}

                                            {result.probability_of_loss && (
                                                <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-lg border border-yellow-200">
                                                    <p className="text-sm font-medium text-yellow-700">Probability of Loss</p>
                                                    <p className="text-xl font-bold text-yellow-800">
                                                        {result.probability_of_loss.toFixed(1)}%
                                                    </p>
                                                </div>
                                            )}

                                            {result.max_loss && result.max_loss !== '∞' && (
                                                <div className="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-lg border border-red-200">
                                                    <p className="text-sm font-medium text-red-700">Maximum Loss</p>
                                                    <p className="text-xl font-bold text-red-800">
                                                        ${Math.abs(result.max_loss).toLocaleString()}
                                                    </p>
                                                </div>
                                            )}

                                            {result.max_gain && result.max_gain !== '∞' && (
                                                <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                                                    <p className="text-sm font-medium text-green-700">Maximum Gain</p>
                                                    <p className="text-xl font-bold text-green-800">
                                                        ${Math.abs(result.max_gain).toLocaleString()}
                                                    </p>
                                                </div>
                                            )}
                                        </div>

                                        {/* Distribution Chart */}
                                        {result.distribution && result.distribution.length > 0 && (
                                            <div className="space-y-4">
                                                <h3 className="text-lg font-semibold text-gray-800">
                                                    📈 Return Distribution
                                                </h3>
                                                <div className="h-80 w-full">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <LineChart
                                                            data={result.distribution.map((v, i) => ({
                                                                index: i,
                                                                value: v,
                                                                return: ((v / result.distribution[0] - 1) * 100).toFixed(2)
                                                            }))}
                                                        >
                                                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                                                            <XAxis
                                                                dataKey="index"
                                                                stroke="#6b7280"
                                                                fontSize={12}
                                                            />
                                                            <YAxis
                                                                stroke="#6b7280"
                                                                fontSize={12}
                                                                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                                                            />
                                                            <Tooltip
                                                                formatter={(value) => [`$${value.toLocaleString()}`, 'Portfolio Value']}
                                                                labelFormatter={(label) => `Scenario ${label}`}
                                                                contentStyle={{
                                                                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                                                                    border: '1px solid #e5e7eb',
                                                                    borderRadius: '8px',
                                                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                                                }}
                                                            />
                                                            <Line
                                                                type="monotone"
                                                                dataKey="value"
                                                                stroke="url(#gradient)"
                                                                strokeWidth={2}
                                                                dot={false}
                                                            />
                                                            <defs>
                                                                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                                                    <stop offset="0%" stopColor="#3b82f6" />
                                                                    <stop offset="100%" stopColor="#6366f1" />
                                                                </linearGradient>
                                                            </defs>
                                                        </LineChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}

                                        {/* Additional Metrics */}
                                        {result.execution_time && (
                                            <div className="bg-gray-50 p-4 rounded-lg">
                                                <p className="text-sm text-gray-600">
                                                    ⏱️ Execution Time: <span className="font-semibold">
                                                        {typeof result.execution_time === 'number' ?
                                                            (result.execution_time > 1 ?
                                                                `${result.execution_time.toFixed(3)}s` :
                                                                `${(result.execution_time * 1000).toFixed(0)}ms`
                                                            ) :
                                                            result.execution_time
                                                        }
                                                    </span>
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </motion.div>
                )}
            </motion.div>
        </div>
    );
}
