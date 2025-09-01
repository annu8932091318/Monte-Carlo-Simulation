/**
 * Node.js Risk Simulation Engine Server
 * 
 * High-performance financial risk modeling API server built with Express.js
 * Supports multiple simulation methods for portfolio risk assessment
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Import and use routes
const routes = require('./routes/routes');
app.use('/', routes);

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        available_endpoints: [
            'GET /health',
            'GET /api/info',
            'GET /api/docs',
            'GET /api/examples',
            'POST /simulate',
            'POST /compare',
            'POST /benchmark'
        ],
        engine: 'node'
    });
});

// Global error handler
app.use((error, req, res, next) => {
    console.error('Server error:', error);
    res.status(500).json({
        error: 'Internal Server Error',
        message: 'An unexpected error occurred',
        engine: 'node'
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Risk Simulation Engine (Node.js) starting on port ${PORT}`);
    console.log(`📊 API Documentation: http://localhost:${PORT}/api/docs`);
    console.log(`🏥 Health Check: http://localhost:${PORT}/health`);
    console.log(`📝 Examples: http://localhost:${PORT}/api/examples`);
    console.log(`🎯 Ready to process risk simulations!`);
});

module.exports = app;

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        engine: 'node',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        version: '1.0.0'
    });
});

// API information endpoint
app.get('/api/info', (req, res) => {
    res.json({
        engine: 'node',
        version: '1.0.0',
        description: 'Risk Simulation Engine - Node.js Implementation',
        supportedMethods: [
            'monteCarlo',
            'advancedMonteCarlo',
            'historicalSimulation',
            'bootstrapHistorical',
            'bootstrap',
            'advancedBootstrap',
            'varianceCovariance',
            'stressTestVarCov',
            'gbm',
            'multiAssetGBM',
            'pathDependentGBM'
        ],
        documentation: '/api/docs',
        examples: '/api/examples'
    });
});

// API documentation endpoint
app.get('/api/docs', (req, res) => {
    res.json({
        title: 'Risk Simulation Engine API',
        description: 'Comprehensive financial risk modeling API with multiple simulation methods',
        baseUrl: `http://localhost:${PORT}`,

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
                    method: 'monteCarlo',
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

        methods: {
            monteCarlo: 'Basic Monte Carlo using portfolio-level GBM',
            advancedMonteCarlo: 'Monte Carlo with explicit asset correlation modeling',
            historicalSimulation: 'Risk assessment using historical return data',
            bootstrapHistorical: 'Bootstrap enhancement of historical simulation',
            bootstrap: 'Bootstrap resampling from historical data',
            advancedBootstrap: 'Block bootstrap with time series preservation',
            varianceCovariance: 'Analytical VaR using normal distribution assumption',
            stressTestVarCov: 'Stress testing using VarCov method',
            gbm: 'Basic Geometric Brownian Motion simulation',
            multiAssetGBM: 'Multi-asset GBM with correlation analysis',
            pathDependentGBM: 'Full path simulation for path-dependent analysis'
        }
    });
});

// Example requests endpoint
app.get('/api/examples', (req, res) => {
    res.json({
        basicMonteCarlo: {
            method: 'POST',
            url: '/simulate',
            body: {
                engine: 'node',
                method: 'monteCarlo',
                portfolio: [100000, 200000, 150000, 50000],
                params: {
                    iterations: 1000000,
                    confidence: 0.95,
                    horizon: 1.0
                }
            }
        },

        historicalSimulation: {
            method: 'POST',
            url: '/simulate',
            body: {
                engine: 'node',
                method: 'historicalSimulation',
                portfolio: [250000, 250000, 250000, 250000],
                params: {
                    confidence: 0.95,
                    horizon: 1,
                    lookbackPeriod: 252
                }
            }
        },

        varianceCovariance: {
            method: 'POST',
            url: '/simulate',
            body: {
                engine: 'node',
                method: 'varianceCovariance',
                portfolio: [300000, 400000, 200000, 100000],
                params: {
                    confidence: 0.95,
                    horizon: 1.0
                }
            }
        },

        methodComparison: {
            method: 'POST',
            url: '/compare',
            body: {
                portfolio: [100000, 100000, 100000, 100000],
                methods: ['monteCarlo', 'historicalSimulation', 'varianceCovariance'],
                params: {
                    confidence: 0.95,
                    horizon: 1.0,
                    iterations: 100000
                }
            }
        }
    });
});

// Main simulation endpoint
app.post('/simulate', async (req, res) => {
    try {
        console.log(`🚀 Received simulation request: ${req.body.method}`);
        const startTime = Date.now();

        const { method, portfolio, params = {} } = req.body;

        // Validate input
        if (!method) {
            return res.status(400).json({
                error: 'Missing required parameter: method',
                engine: 'node'
            });
        }

        if (!portfolio || !Array.isArray(portfolio) || portfolio.length === 0) {
            return res.status(400).json({
                error: 'Invalid portfolio: must be non-empty array of numbers',
                engine: 'node'
            });
        }

        // Validate portfolio values
        if (portfolio.some(val => typeof val !== 'number' || val <= 0)) {
            return res.status(400).json({
                error: 'Invalid portfolio values: all values must be positive numbers',
                engine: 'node'
            });
        }

        let result;

        // Route to appropriate simulation method
        switch (method) {
            case 'monteCarlo':
                result = monteCarloSimulation(portfolio, params);
                break;

            case 'advancedMonteCarlo':
                result = advancedMonteCarloSimulation(portfolio, params);
                break;

            case 'historicalSimulation':
                result = historicalSimulation(portfolio, params);
                break;

            case 'bootstrapHistorical':
                result = bootstrapHistoricalSimulation(portfolio, params);
                break;

            case 'bootstrap':
                result = bootstrapSimulation(portfolio, params);
                break;

            case 'advancedBootstrap':
                result = advancedBlockBootstrap(portfolio, params);
                break;

            case 'varianceCovariance':
                result = varianceCovarianceVaR(portfolio, params);
                break;

            case 'stressTestVarCov':
                result = stressTestVarCov(portfolio, params);
                break;

            case 'gbm':
                result = geometricBrownianMotion(portfolio, params);
                break;

            case 'multiAssetGBM':
                result = multiAssetGBM(portfolio, params);
                break;

            case 'pathDependentGBM':
                result = pathDependentGBM(portfolio, params);
                break;

            default:
                return res.status(400).json({
                    error: `Unknown simulation method: ${method}`,
                    supportedMethods: [
                        'monteCarlo', 'advancedMonteCarlo', 'historicalSimulation',
                        'bootstrapHistorical', 'bootstrap', 'advancedBootstrap',
                        'varianceCovariance', 'stressTestVarCov', 'gbm',
                        'multiAssetGBM', 'pathDependentGBM'
                    ],
                    engine: 'node'
                });
        }

        const totalTime = (Date.now() - startTime) / 1000;

        // Add server metadata
        const response = {
            ...result,
            serverInfo: {
                engine: 'node',
                method,
                requestTime: new Date().toISOString(),
                totalProcessingTime: totalTime,
                nodeVersion: process.version,
                memoryUsage: process.memoryUsage()
            }
        };

        console.log(`✅ Simulation completed in ${totalTime.toFixed(3)}s`);
        res.json(response);

    } catch (error) {
        console.error('Simulation error:', error);
        res.status(500).json({
            error: 'Simulation failed',
            message: error.message,
            engine: 'node'
        });
    }
});

// Method comparison endpoint
app.post('/compare', async (req, res) => {
    try {
        console.log('🔍 Running method comparison...');
        const startTime = Date.now();

        const { portfolio, methods, params = {} } = req.body;

        // Validate input
        if (!portfolio || !Array.isArray(portfolio)) {
            return res.status(400).json({
                error: 'Invalid portfolio array',
                engine: 'node'
            });
        }

        if (!methods || !Array.isArray(methods)) {
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

                // Create a request object for the simulate endpoint
                const simulateRequest = {
                    body: { method, portfolio, params }
                };

                // Use a mock response object to capture the result
                let methodResult;
                const mockRes = {
                    json: (data) => { methodResult = data; },
                    status: () => mockRes
                };

                // Call the simulate endpoint logic
                await new Promise((resolve, reject) => {
                    // Simulate the endpoint call
                    setTimeout(() => {
                        try {
                            switch (method) {
                                case 'monteCarlo':
                                    methodResult = monteCarloSimulation(portfolio, params);
                                    break;
                                case 'historicalSimulation':
                                    methodResult = historicalSimulation(portfolio, params);
                                    break;
                                case 'varianceCovariance':
                                    methodResult = varianceCovarianceVaR(portfolio, params);
                                    break;
                                case 'gbm':
                                    methodResult = geometricBrownianMotion(portfolio, params);
                                    break;
                                default:
                                    throw new Error(`Unsupported comparison method: ${method}`);
                            }
                            resolve();
                        } catch (err) {
                            reject(err);
                        }
                    }, 0);
                });

                const methodTime = (Date.now() - methodStartTime) / 1000;

                results[method] = {
                    riskMetrics: methodResult.riskMetrics,
                    portfolioStats: methodResult.portfolioStats,
                    performance: methodResult.performance || { simulationTime: methodTime }
                };

                timings[method] = methodTime;

            } catch (error) {
                console.error(`Error running ${method}:`, error);
                results[method] = {
                    error: error.message
                };
                timings[method] = 0;
            }
        }

        // Calculate comparison metrics
        const varComparison = {};
        const cvarComparison = {};
        const performanceComparison = {};

        methods.forEach(method => {
            if (results[method].riskMetrics) {
                varComparison[method] = results[method].riskMetrics.VaR_95;
                cvarComparison[method] = results[method].riskMetrics.CVaR_95;
                performanceComparison[method] = timings[method];
            }
        });

        const totalTime = (Date.now() - startTime) / 1000;

        res.json({
            comparison: {
                portfolio,
                methods,
                params,
                results,
                analysis: {
                    varComparison,
                    cvarComparison,
                    performanceComparison,
                    fastestMethod: Object.entries(performanceComparison)
                        .reduce((a, b) => performanceComparison[a[0]] < performanceComparison[b[0]] ? a : b)[0],
                    mostConservativeVaR: Object.entries(varComparison)
                        .reduce((a, b) => varComparison[a[0]] > varComparison[b[0]] ? a : b)[0]
                }
            },
            metadata: {
                engine: 'node',
                totalProcessingTime: totalTime,
                timestamp: new Date().toISOString()
            }
        });

        console.log(`✅ Method comparison completed in ${totalTime.toFixed(3)}s`);

    } catch (error) {
        console.error('Comparison error:', error);
        res.status(500).json({
            error: 'Comparison failed',
            message: error.message,
            engine: 'node'
        });
    }
});

// Performance benchmark endpoint
app.post('/benchmark', async (req, res) => {
    try {
        console.log('🏃 Running performance benchmark...');
        const startTime = Date.now();

        const {
            portfolio,
            method = 'monteCarlo',
            iterations = [10000, 50000, 100000, 500000, 1000000],
            params = {}
        } = req.body;

        const benchmarkResults = [];

        for (const iterCount of iterations) {
            console.log(`  Benchmarking ${method} with ${iterCount.toLocaleString()} iterations...`);

            const benchParams = { ...params, iterations: iterCount };
            const benchStartTime = Date.now();

            let result;
            switch (method) {
                case 'monteCarlo':
                    result = monteCarloSimulation(portfolio, benchParams);
                    break;
                case 'gbm':
                    result = geometricBrownianMotion(portfolio, benchParams);
                    break;
                default:
                    throw new Error(`Benchmarking not supported for method: ${method}`);
            }

            const benchTime = (Date.now() - benchStartTime) / 1000;
            const throughput = iterCount / benchTime;

            benchmarkResults.push({
                iterations: iterCount,
                time: benchTime,
                throughput: Math.round(throughput),
                VaR_95: result.riskMetrics.VaR_95,
                CVaR_95: result.riskMetrics.CVaR_95
            });
        }

        const totalTime = (Date.now() - startTime) / 1000;

        res.json({
            benchmark: {
                method,
                portfolio,
                results: benchmarkResults,
                analysis: {
                    maxThroughput: Math.max(...benchmarkResults.map(r => r.throughput)),
                    avgThroughput: benchmarkResults.reduce((sum, r) => sum + r.throughput, 0) / benchmarkResults.length,
                    scalingEfficiency: benchmarkResults[benchmarkResults.length - 1].throughput / benchmarkResults[0].throughput
                }
            },
            metadata: {
                engine: 'node',
                totalBenchmarkTime: totalTime,
                timestamp: new Date().toISOString()
            }
        });

        console.log(`✅ Benchmark completed in ${totalTime.toFixed(3)}s`);

    } catch (error) {
        console.error('Benchmark error:', error);
        res.status(500).json({
            error: 'Benchmark failed',
            message: error.message,
            engine: 'node'
        });
    }
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        availableEndpoints: [
            'GET /health',
            'GET /api/info',
            'GET /api/docs',
            'GET /api/examples',
            'POST /simulate',
            'POST /compare',
            'POST /benchmark'
        ],
        engine: 'node'
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Risk Simulation Engine (Node.js) running on port ${PORT}`);
    console.log(`📊 API Documentation: http://localhost:${PORT}/api/docs`);
    console.log(`🏥 Health Check: http://localhost:${PORT}/health`);
    console.log(`📝 Examples: http://localhost:${PORT}/api/examples`);
    console.log(`🎯 Ready to process risk simulations!`);
});

module.exports = app;
