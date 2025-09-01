/**
 * Monte Carlo Risk Simulation using Geometric Brownian Motion
 * 
 * This module implements Monte Carlo simulation for portfolio risk assessment
 * using the Geometric Brownian Motion model. It generates thousands of possible
 * future portfolio values and calculates risk metrics.
 */

const {
    randomNormalArray,
    calculateVaR,
    calculateCVaR,
    calculateStatistics,
    calculatePortfolioStats,
    correlationToCovariance,
    generateCorrelatedNormals
} = require('../utils/mathHelpers');

/**
 * Monte Carlo simulation for portfolio risk using GBM
 * 
 * Mathematical Foundation:
 * Portfolio value follows: S(T) = S(0) * exp((μ - σ²/2) * T + σ * √T * Z)
 * Where Z ~ N(0,1) is standard normal random variable
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - Simulation parameters
 * @returns {Object} Risk metrics and simulation results
 */
function monteCarloSimulation(portfolio, params = {}) {
    const {
        iterations = 1000000,
        confidence = 0.95,
        horizon = 1.0,           // Time horizon in years
        expectedReturns = [0.08, 0.05, 0.12, 0.03],  // Annual expected returns
        volatilities = [0.20, 0.15, 0.25, 0.10],     // Annual volatilities
        correlationMatrix = [    // Correlation matrix
            [1.00, 0.40, 0.30, 0.10],
            [0.40, 1.00, 0.35, 0.15],
            [0.30, 0.35, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00]
        ]
    } = params;

    // Calculate initial portfolio value
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);

    // Calculate portfolio weights
    const weights = portfolio.map(value => value / initialValue);

    // Convert correlation to covariance matrix
    const covarianceMatrix = correlationToCovariance(volatilities, correlationMatrix);

    // Calculate portfolio-level parameters
    const portfolioStats = calculatePortfolioStats(weights, expectedReturns, covarianceMatrix);
    const { expectedReturn: mu_p, volatility: sigma_p } = portfolioStats;

    console.log(`🎲 Starting Monte Carlo simulation with ${iterations.toLocaleString()} iterations...`);
    const startTime = Date.now();

    // Pre-calculate GBM parameters
    const driftTerm = (mu_p - 0.5 * sigma_p * sigma_p) * horizon;
    const diffusionStd = sigma_p * Math.sqrt(horizon);

    // Generate random shocks and calculate terminal values
    const terminalValues = [];
    const losses = [];

    for (let i = 0; i < iterations; i++) {
        // Generate standard normal random variable
        const Z = randomNormalArray(1, 0, 1)[0];

        // Apply GBM formula
        const terminalValue = initialValue * Math.exp(driftTerm + diffusionStd * Z);
        terminalValues.push(terminalValue);

        // Calculate loss (positive = loss, negative = gain)
        losses.push(initialValue - terminalValue);
    }

    const simulationTime = (Date.now() - startTime) / 1000;

    // Calculate risk metrics
    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Calculate statistics
    const terminalStats = calculateStatistics(terminalValues);
    const lossStats = calculateStatistics(losses);

    // Calculate probability of loss
    const lossCount = terminalValues.filter(val => val < initialValue).length;
    const probabilityOfLoss = lossCount / iterations;

    // Calculate percentiles
    const sortedTerminals = [...terminalValues].sort((a, b) => a - b);
    const p10 = sortedTerminals[Math.floor(0.10 * iterations)];
    const p50 = sortedTerminals[Math.floor(0.50 * iterations)];
    const p90 = sortedTerminals[Math.floor(0.90 * iterations)];

    return {
        // Simulation Configuration
        simulationConfig: {
            method: 'monteCarlo',
            engine: 'node',
            iterations,
            horizon,
            confidence,
            initialValue,
            portfolioMu: mu_p,
            portfolioSigma: sigma_p
        },

        // Risk Metrics
        riskMetrics: {
            VaR_95: VaR,
            CVaR_95: CVaR,
            probabilityOfLoss: probabilityOfLoss * 100,  // As percentage
            expectedShortfall: CVaR
        },

        // Portfolio Statistics
        portfolioStats: {
            expectedTerminal: terminalStats.mean,
            standardDeviation: terminalStats.std,
            expectedReturn: ((terminalStats.mean / initialValue) - 1) * 100,  // As percentage
            volatility: sigma_p * 100  // As percentage
        },

        // Scenario Analysis
        scenarios: {
            pessimistic_p10: p10,
            median_p50: p50,
            optimistic_p90: p90
        },

        // Performance Metrics
        performance: {
            simulationTime: simulationTime,
            iterationsPerSecond: Math.round(iterations / simulationTime)
        },

        // Optional: Distribution data for plotting (first 10000 points to avoid large payload)
        distribution: terminalValues.slice(0, Math.min(10000, iterations))
    };
}

/**
 * Advanced Monte Carlo with correlation modeling
 * 
 * This version explicitly models asset correlations and generates
 * correlated return paths for each asset individually.
 * 
 * @param {Array<number>} portfolio - Portfolio asset values  
 * @param {Object} params - Advanced simulation parameters
 * @returns {Object} Detailed risk metrics with asset-level analysis
 */
function advancedMonteCarloSimulation(portfolio, params = {}) {
    const {
        iterations = 1000000,
        confidence = 0.95,
        horizon = 1.0,
        expectedReturns = [0.08, 0.05, 0.12, 0.03],
        volatilities = [0.20, 0.15, 0.25, 0.10],
        correlationMatrix = [
            [1.00, 0.40, 0.30, 0.10],
            [0.40, 1.00, 0.35, 0.15],
            [0.30, 0.35, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00]
        ]
    } = params;

    const numAssets = portfolio.length;
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);

    console.log(`🎯 Starting Advanced Monte Carlo simulation with ${iterations.toLocaleString()} iterations...`);
    const startTime = Date.now();

    // Convert correlation to covariance matrix
    const covarianceMatrix = correlationToCovariance(volatilities, correlationMatrix);

    // Generate correlated random returns for all assets
    const correlatedReturns = generateCorrelatedNormals(iterations, covarianceMatrix);

    const terminalValues = [];
    const assetPaths = Array(numAssets).fill().map(() => []);

    for (let i = 0; i < iterations; i++) {
        let portfolioValue = 0;

        for (let j = 0; j < numAssets; j++) {
            // Apply GBM to each asset individually
            const driftTerm = (expectedReturns[j] - 0.5 * volatilities[j] * volatilities[j]) * horizon;
            const diffusionTerm = volatilities[j] * Math.sqrt(horizon) * correlatedReturns[i][j];

            const assetTerminalValue = portfolio[j] * Math.exp(driftTerm + diffusionTerm);
            assetPaths[j].push(assetTerminalValue);
            portfolioValue += assetTerminalValue;
        }

        terminalValues.push(portfolioValue);
    }

    const simulationTime = (Date.now() - startTime) / 1000;

    // Calculate losses and risk metrics
    const losses = terminalValues.map(val => initialValue - val);
    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Calculate asset-level statistics
    const assetStats = assetPaths.map((path, index) => {
        const stats = calculateStatistics(path);
        return {
            assetIndex: index,
            initialValue: portfolio[index],
            expectedTerminal: stats.mean,
            volatility: stats.std,
            expectedReturn: ((stats.mean / portfolio[index]) - 1) * 100
        };
    });

    return {
        // Basic metrics (same structure as simple Monte Carlo)
        ...monteCarloSimulation(portfolio, params),

        // Advanced asset-level analysis
        assetAnalysis: assetStats,

        // Correlation impact
        correlationImpact: {
            averageCorrelation: correlationMatrix.flat()
                .filter((val, idx) => idx % (numAssets + 1) !== 0)  // Exclude diagonal
                .reduce((sum, val) => sum + val, 0) / (numAssets * (numAssets - 1)),

            diversificationBenefit: {
                description: "Portfolio volatility vs weighted average of individual volatilities",
                portfolioVolatility: Math.sqrt(portfolio.reduce((sum, weight, i) =>
                    sum + Math.pow(weight / initialValue, 2) * Math.pow(volatilities[i], 2), 0)),
                weightedAverageVolatility: portfolio.reduce((sum, weight, i) =>
                    sum + (weight / initialValue) * volatilities[i], 0)
            }
        }
    };
}

module.exports = {
    monteCarloSimulation,
    advancedMonteCarloSimulation
};
