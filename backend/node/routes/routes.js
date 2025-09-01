/**
 * Node.js Backend Routes
 * 
 * This file contains all API routes for the Node.js risk simulation engine.
 * Routes are organized by functionality and include comprehensive error handling.
 */

const express = require('express');
const router = express.Router();

// Import simulation modules
const { monteCarloSimulation, advancedMonteCarloSimulation } = require('../simulations/monteCarlo');
const { historicalSimulation, bootstrapHistoricalSimulation } = require('../simulations/historical');
const { bootstrapSimulation, advancedBootstrapSimulation } = require('../simulations/bootstrap');
const { varianceCovarianceMethod, stressTestVarCov } = require('../simulations/varcov');
const { gbmSimulation, multiAssetGBM, pathDependentGBM } = require('../simulations/gbm');

// Utility functions
function validatePortfolio(portfolio) {
    if (!Array.isArray(portfolio)) {
        return { valid: false, message: 'Portfolio must be an array' };
    }
    if (portfolio.length === 0) {
        return { valid: false, message: 'Portfolio cannot be empty' };
    }
    if (!portfolio.every(value => typeof value === 'number' && value > 0)) {
        return { valid: false, message: 'Portfolio values must be positive numbers' };
    }
    return { valid: true };
}

function handleSimulationError(error) {
    console.error('Simulation error:', error);
    return {
        error: 'Simulation failed',
        message: error.message,
        type: error.constructor.name,
        engine: 'node'
    };
}

// Health check route
router.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        engine: 'node',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        node_version: process.version,
        uptime: process.uptime(),
        memory_usage: process.memoryUsage()
    });
});

// API information route
router.get('/api/info', (req, res) => {
    res.json({
        engine: 'node',
        version: '1.0.0',
        description: 'Risk Simulation Engine - Node.js Implementation',
        supported_methods: [
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
        documentation: '/api/docs',
        examples: '/api/examples',
        technology_stack: {
            language: 'JavaScript',
            runtime: 'Node.js',
            framework: 'Express.js',
            numerical_computing: 'MathJS + Custom implementations',
            performance: 'Optimized JavaScript with V8'
        }
    });
});

// API documentation route
router.get('/api/docs', (req, res) => {
    res.json({
        title: 'Risk Simulation Engine API - Node.js',
        description: 'High-performance financial risk modeling API implemented in Node.js',
        base_url: `http://localhost:${process.env.PORT || 3001}`,

        endpoints: {
            '/simulate': {
                method: 'POST',
                description: 'Run risk simulation with specified method',
                parameters: {
                    engine: 'string (always "node" for this server)',
                    method: 'string (simulation method to use)',
                    portfolio: 'array of numbers (asset values)',
                    params: 'object (method-specific parameters)'
                },
                example: {
                    engine: 'node',
                    method: 'monte_carlo',
                    portfolio: [100000, 200000, 150000, 50000],
                    params: {
                        iterations: 1000000,
                        confidence: 0.95,
                        horizon: 1.0
                    }
                }
            },

            '/compare': {
                method: 'POST',
                description: 'Compare multiple simulation methods',
                parameters: {
                    portfolio: 'array of numbers',
                    methods: 'array of method names',
                    params: 'object (shared parameters)'
                }
            },

            '/benchmark': {
                method: 'POST',
                description: 'Performance benchmark of simulation methods',
                parameters: {
                    portfolio: 'array of numbers',
                    iterations: 'array of iteration counts',
                    method: 'string (method to benchmark)'
                }
            }
        },

        node_advantages: {
            performance: 'V8 JavaScript engine optimization',
            concurrency: 'Event-driven non-blocking I/O',
            ecosystem: 'Rich npm package ecosystem',
            real_time: 'Excellent for real-time applications',
            web_integration: 'Native web and API development'
        }
    });
});

// API examples route
router.get('/api/examples', (req, res) => {
    res.json({
        basic_monte_carlo: {
            method: 'POST',
            url: '/simulate',
            body: {
                engine: 'node',
                method: 'monte_carlo',
                portfolio: [100000, 200000, 150000, 50000],
                params: {
                    iterations: 1000000,
                    confidence: 0.95,
                    horizon: 1.0
                }
            },
            description: 'Basic Monte Carlo simulation using portfolio-level GBM'
        },

        historical_simulation: {
            method: 'POST',
            url: '/simulate',
            body: {
                engine: 'node',
                method: 'historical_simulation',
                portfolio: [250000, 250000, 250000, 250000],
                params: {
                    iterations: 100000,
                    confidence: 0.95,
                    horizon: 1.0,
                    lookback_periods: 252
                }
            },
            description: 'Historical simulation using past market data patterns'
        },

        variance_covariance: {
            method: 'POST',
            url: '/simulate',
            body: {
                engine: 'node',
                method: 'variance_covariance',
                portfolio: [100000, 100000, 100000, 100000],
                params: {
                    confidence: 0.95,
                    horizon: 1.0,
                    expected_returns: [0.08, 0.05, 0.12, 0.03],
                    volatilities: [0.20, 0.15, 0.25, 0.10],
                    correlations: 0.3
                }
            },
            description: 'Analytical variance-covariance method for fast risk calculation'
        },

        method_comparison: {
            method: 'POST',
            url: '/compare',
            body: {
                portfolio: [100000, 100000, 100000, 100000],
                methods: ['monte_carlo', 'variance_covariance', 'historical_simulation'],
                params: {
                    confidence: 0.95,
                    horizon: 1.0,
                    iterations: 100000
                }
            },
            description: 'Compare different simulation methods side by side'
        }
    });
});

// Main simulation route
router.post('/simulate', async (req, res) => {
    try {
        console.log(`🚀 Received simulation request: ${req.body.method || 'unknown'}`);
        const startTime = Date.now();

        const { method, portfolio, params = {} } = req.body;

        // Validate input
        if (!method) {
            return res.status(400).json({
                error: 'Missing required parameter: method',
                engine: 'node'
            });
        }

        const portfolioValidation = validatePortfolio(portfolio);
        if (!portfolioValidation.valid) {
            return res.status(400).json({
                error: 'Invalid portfolio',
                message: portfolioValidation.message,
                engine: 'node'
            });
        }

        let result;

        // Route to appropriate simulation method
        switch (method) {
            case 'monte_carlo':
                result = monteCarloSimulation(portfolio, params);
                break;
            case 'advanced_monte_carlo':
                result = advancedMonteCarloSimulation(portfolio, params);
                break;
            case 'historical_simulation':
                result = historicalSimulation(portfolio, params);
                break;
            case 'bootstrap_historical':
                result = bootstrapHistoricalSimulation(portfolio, params);
                break;
            case 'bootstrap':
                result = bootstrapSimulation(portfolio, params);
                break;
            case 'advanced_bootstrap':
                result = advancedBootstrapSimulation(portfolio, params);
                break;
            case 'variance_covariance':
                result = varianceCovarianceMethod(portfolio, params);
                break;
            case 'stress_test_varcov':
                result = stressTestVarCov(portfolio, params);
                break;
            case 'gbm':
                result = gbmSimulation(portfolio, params);
                break;
            case 'multi_asset_gbm':
                result = multiAssetGBM(portfolio, params);
                break;
            case 'path_dependent_gbm':
                result = pathDependentGBM(portfolio, params);
                break;
            default:
                return res.status(400).json({
                    error: `Unknown simulation method: ${method}`,
                    supported_methods: [
                        'monte_carlo', 'advanced_monte_carlo', 'historical_simulation',
                        'bootstrap_historical', 'bootstrap', 'advanced_bootstrap',
                        'variance_covariance', 'stress_test_varcov', 'gbm',
                        'multi_asset_gbm', 'path_dependent_gbm'
                    ],
                    engine: 'node'
                });
        }

        const totalTime = Date.now() - startTime;

        // Add server metadata
        result.server_info = {
            engine: 'node',
            method: method,
            request_time: new Date().toISOString(),
            total_processing_time: totalTime / 1000,
            node_version: process.version,
            v8_version: process.versions.v8
        };

        console.log(`✅ Simulation completed in ${totalTime}ms`);
        res.json(result);

    } catch (error) {
        console.error('❌ Simulation error:', error);
        res.status(500).json(handleSimulationError(error));
    }
});

// Method comparison route
router.post('/compare', async (req, res) => {
    try {
        console.log('🔍 Running method comparison...');
        const startTime = Date.now();

        const { portfolio, methods, params = {} } = req.body;

        // Validate input
        const portfolioValidation = validatePortfolio(portfolio);
        if (!portfolioValidation.valid) {
            return res.status(400).json({
                error: 'Invalid portfolio',
                message: portfolioValidation.message,
                engine: 'node'
            });
        }

        if (!Array.isArray(methods) || methods.length === 0) {
            return res.status(400).json({
                error: 'Invalid methods array',
                engine: 'node'
            });
        }

        const results = {};
        const timings = {};

        // Run each method
        for (const method of methods) {
            try {
                console.log(`  Running ${method}...`);
                const methodStartTime = Date.now();

                let methodResult;

                switch (method) {
                    case 'monte_carlo':
                        methodResult = monteCarloSimulation(portfolio, params);
                        break;
                    case 'advanced_monte_carlo':
                        methodResult = advancedMonteCarloSimulation(portfolio, params);
                        break;
                    case 'historical_simulation':
                        methodResult = historicalSimulation(portfolio, params);
                        break;
                    case 'bootstrap_historical':
                        methodResult = bootstrapHistoricalSimulation(portfolio, params);
                        break;
                    case 'bootstrap':
                        methodResult = bootstrapSimulation(portfolio, params);
                        break;
                    case 'advanced_bootstrap':
                        methodResult = advancedBootstrapSimulation(portfolio, params);
                        break;
                    case 'variance_covariance':
                        methodResult = varianceCovarianceMethod(portfolio, params);
                        break;
                    case 'stress_test_varcov':
                        methodResult = stressTestVarCov(portfolio, params);
                        break;
                    case 'gbm':
                        methodResult = gbmSimulation(portfolio, params);
                        break;
                    case 'multi_asset_gbm':
                        methodResult = multiAssetGBM(portfolio, params);
                        break;
                    case 'path_dependent_gbm':
                        methodResult = pathDependentGBM(portfolio, params);
                        break;
                    default:
                        results[method] = { error: `Unsupported comparison method: ${method}` };
                        timings[method] = 0;
                        continue;
                }

                const methodTime = Date.now() - methodStartTime;

                results[method] = {
                    risk_metrics: methodResult.risk_metrics,
                    portfolio_stats: methodResult.portfolio_stats,
                    performance: methodResult.performance || { simulation_time: methodTime / 1000 }
                };

                timings[method] = methodTime / 1000;

            } catch (error) {
                console.error(`Error running ${method}:`, error);
                results[method] = { error: error.message };
                timings[method] = 0;
            }
        }

        // Calculate comparison metrics
        const varComparison = {};
        const cvarComparison = {};
        const performanceComparison = {};

        for (const method of methods) {
            if (results[method].risk_metrics) {
                varComparison[method] = results[method].risk_metrics.VaR_95;
                cvarComparison[method] = results[method].risk_metrics.CVaR_95;
                performanceComparison[method] = timings[method];
            }
        }

        const totalTime = Date.now() - startTime;

        const response = {
            comparison: {
                portfolio,
                methods,
                params,
                results,
                analysis: {
                    var_comparison: varComparison,
                    cvar_comparison: cvarComparison,
                    performance_comparison: performanceComparison
                }
            },
            metadata: {
                engine: 'node',
                total_processing_time: totalTime / 1000,
                timestamp: new Date().toISOString()
            }
        };

        // Add fastest method if we have performance data
        if (Object.keys(performanceComparison).length > 0) {
            const fastestMethod = Object.entries(performanceComparison)
                .reduce((a, b) => a[1] < b[1] ? a : b)[0];
            response.comparison.analysis.fastest_method = fastestMethod;
        }

        console.log(`✅ Method comparison completed in ${totalTime}ms`);
        res.json(response);

    } catch (error) {
        console.error('❌ Comparison error:', error);
        res.status(500).json(handleSimulationError(error));
    }
});

// Performance benchmark route
router.post('/benchmark', async (req, res) => {
    try {
        console.log('🏃 Running performance benchmark...');
        const startTime = Date.now();

        const {
            portfolio,
            method = 'monte_carlo',
            iterations = [10000, 50000, 100000, 500000, 1000000],
            params = {}
        } = req.body;

        const portfolioValidation = validatePortfolio(portfolio);
        if (!portfolioValidation.valid) {
            return res.status(400).json({
                error: 'Invalid portfolio',
                message: portfolioValidation.message,
                engine: 'node'
            });
        }

        const benchmarkResults = [];

        for (const iterCount of iterations) {
            console.log(`  Benchmarking ${method} with ${iterCount.toLocaleString()} iterations...`);

            const benchParams = { ...params, iterations: iterCount };
            const benchStartTime = Date.now();

            let result;

            switch (method) {
                case 'monte_carlo':
                    result = monteCarloSimulation(portfolio, benchParams);
                    break;
                case 'advanced_monte_carlo':
                    result = advancedMonteCarloSimulation(portfolio, benchParams);
                    break;
                case 'historical_simulation':
                    result = historicalSimulation(portfolio, benchParams);
                    break;
                case 'bootstrap_historical':
                    result = bootstrapHistoricalSimulation(portfolio, benchParams);
                    break;
                case 'bootstrap':
                    result = bootstrapSimulation(portfolio, benchParams);
                    break;
                case 'advanced_bootstrap':
                    result = advancedBootstrapSimulation(portfolio, benchParams);
                    break;
                case 'variance_covariance':
                    result = varianceCovarianceMethod(portfolio, benchParams);
                    break;
                case 'gbm':
                    result = gbmSimulation(portfolio, benchParams);
                    break;
                case 'multi_asset_gbm':
                    result = multiAssetGBM(portfolio, benchParams);
                    break;
                case 'path_dependent_gbm':
                    result = pathDependentGBM(portfolio, benchParams);
                    break;
                default:
                    return res.status(400).json({
                        error: `Benchmarking not supported for method: ${method}`,
                        engine: 'node'
                    });
            }

            const benchTime = Date.now() - benchStartTime;
            const throughput = benchTime > 0 ? Math.round(iterCount / (benchTime / 1000)) : 0;

            benchmarkResults.push({
                iterations: iterCount,
                time: benchTime / 1000,
                throughput,
                VaR_95: result.risk_metrics.VaR_95,
                CVaR_95: result.risk_metrics.CVaR_95
            });
        }

        const totalTime = Date.now() - startTime;

        // Calculate analysis metrics
        const throughputs = benchmarkResults.map(r => r.throughput).filter(t => t > 0);

        const response = {
            benchmark: {
                method,
                portfolio,
                results: benchmarkResults,
                analysis: {
                    max_throughput: throughputs.length > 0 ? Math.max(...throughputs) : 0,
                    avg_throughput: throughputs.length > 0 ? Math.round(throughputs.reduce((a, b) => a + b) / throughputs.length) : 0,
                    scaling_efficiency: throughputs.length >= 2 ? throughputs[throughputs.length - 1] / throughputs[0] : 1.0
                }
            },
            metadata: {
                engine: 'node',
                total_benchmark_time: totalTime / 1000,
                timestamp: new Date().toISOString()
            }
        };

        console.log(`✅ Benchmark completed in ${totalTime}ms`);
        res.json(response);

    } catch (error) {
        console.error('❌ Benchmark error:', error);
        res.status(500).json(handleSimulationError(error));
    }
});

// Error handling middleware for routes
router.use((error, req, res, next) => {
    console.error('Route error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message,
        engine: 'node'
    });
});

module.exports = router;
