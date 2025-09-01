/**
 * Geometric Brownian Motion (GBM) Simulation
 * 
 * This module implements various GBM-based simulation strategies for portfolio
 * risk assessment, including basic GBM, multi-asset GBM with correlations,
 * and path-dependent analysis.
 */

const {
    randomNormalArray,
    generateCorrelatedNormals,
    calculateVaR,
    calculateCVaR,
    calculateStatistics,
    calculatePortfolioStats,
    correlationToCovariance
} = require('../utils/mathHelpers');

/**
 * Basic Geometric Brownian Motion simulation
 * 
 * Models asset prices using: dS = μS dt + σS dW
 * Analytical solution: S(T) = S(0) * exp((μ - σ²/2)T + σ√T * Z)
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - GBM simulation parameters
 * @returns {Object} GBM simulation results
 */
function geometricBrownianMotion(portfolio, params = {}) {
    const {
        iterations = 1000000,
        confidence = 0.95,
        horizon = 1.0,                  // Time horizon in years
        expectedReturns = [0.08, 0.05, 0.12, 0.03],  // Annual expected returns (drift)
        volatilities = [0.20, 0.15, 0.25, 0.10],     // Annual volatilities
        correlationMatrix = [
            [1.00, 0.40, 0.30, 0.10],
            [0.40, 1.00, 0.35, 0.15],
            [0.30, 0.35, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00]
        ]
    } = params;

    console.log(`📈 Starting GBM simulation with ${iterations.toLocaleString()} paths...`);
    const startTime = Date.now();

    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);
    const weights = portfolio.map(value => value / initialValue);

    // Convert correlation to covariance matrix
    const covarianceMatrix = correlationToCovariance(volatilities, correlationMatrix);

    // Calculate portfolio-level parameters
    const portfolioStats = calculatePortfolioStats(weights, expectedReturns, covarianceMatrix);
    const { expectedReturn: mu_p, volatility: sigma_p } = portfolioStats;

    // Pre-calculate GBM parameters
    const driftTerm = (mu_p - 0.5 * sigma_p * sigma_p) * horizon;
    const diffusionStd = sigma_p * Math.sqrt(horizon);

    // Generate terminal values using GBM analytical solution
    const terminalValues = [];
    const randomShocks = randomNormalArray(iterations, 0, 1);

    for (let i = 0; i < iterations; i++) {
        const terminalValue = initialValue * Math.exp(driftTerm + diffusionStd * randomShocks[i]);
        terminalValues.push(terminalValue);
    }

    const simulationTime = (Date.now() - startTime) / 1000;

    // Calculate risk metrics
    const losses = terminalValues.map(val => initialValue - val);
    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Calculate statistics
    const terminalStats = calculateStatistics(terminalValues);

    // Calculate probability of loss
    const lossCount = terminalValues.filter(val => val < initialValue).length;
    const probabilityOfLoss = lossCount / iterations;

    return {
        // Simulation Configuration
        simulationConfig: {
            method: 'geometricBrownianMotion',
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
            probabilityOfLoss: probabilityOfLoss * 100,
            expectedShortfall: CVaR
        },

        // Portfolio Statistics
        portfolioStats: {
            expectedTerminal: terminalStats.mean,
            standardDeviation: terminalStats.std,
            expectedReturn: ((terminalStats.mean / initialValue) - 1) * 100,
            volatility: sigma_p * 100
        },

        // GBM Model Properties
        gbmProperties: {
            logNormalDistribution: true,
            meanRevertingNote: 'No mean reversion (pure drift)',
            jumpRiskNote: 'No jump risk modeled',
            correlationModeled: true,
            pathIndependent: true
        },

        // Performance Metrics
        performance: {
            simulationTime: simulationTime,
            pathsPerSecond: Math.round(iterations / simulationTime)
        },

        // Distribution (subset for plotting)
        distribution: terminalValues.slice(0, Math.min(10000, iterations))
    };
}

/**
 * Multi-asset GBM with explicit correlation modeling
 * 
 * Simulates each asset individually and then combines them into portfolio value.
 * This provides asset-level analysis and better correlation modeling.
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - Multi-asset GBM parameters
 * @returns {Object} Multi-asset GBM results with asset-level analysis
 */
function multiAssetGBM(portfolio, params = {}) {
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

    console.log(`🎯 Starting Multi-Asset GBM simulation with ${iterations.toLocaleString()} paths...`);
    const startTime = Date.now();

    const numAssets = portfolio.length;
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);

    // Convert correlation to covariance matrix for random number generation
    const correlationCovMatrix = correlationMatrix.map(row => [...row]);  // Use correlation as covariance for normal generation

    // Generate correlated random shocks for all assets
    const correlatedShocks = generateCorrelatedNormals(iterations, correlationCovMatrix);

    // Simulate each asset individually
    const assetPaths = Array(numAssets).fill().map(() => []);
    const terminalValues = [];

    for (let i = 0; i < iterations; i++) {
        let portfolioTerminalValue = 0;

        for (let asset = 0; asset < numAssets; asset++) {
            // Apply GBM formula to each asset
            const driftTerm = (expectedReturns[asset] - 0.5 * volatilities[asset] * volatilities[asset]) * horizon;
            const diffusionTerm = volatilities[asset] * Math.sqrt(horizon) * correlatedShocks[i][asset];

            const assetTerminalValue = portfolio[asset] * Math.exp(driftTerm + diffusionTerm);
            assetPaths[asset].push(assetTerminalValue);
            portfolioTerminalValue += assetTerminalValue;
        }

        terminalValues.push(portfolioTerminalValue);
    }

    const simulationTime = (Date.now() - startTime) / 1000;

    // Calculate portfolio-level risk metrics
    const losses = terminalValues.map(val => initialValue - val);
    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Calculate asset-level statistics
    const assetAnalysis = assetPaths.map((path, index) => {
        const stats = calculateStatistics(path);
        const assetLosses = path.map(val => portfolio[index] - val);
        const assetVaR = calculateVaR(assetLosses, confidence);

        return {
            assetIndex: index,
            initialValue: portfolio[index],
            expectedTerminal: stats.mean,
            standardDeviation: stats.std,
            expectedReturn: ((stats.mean / portfolio[index]) - 1) * 100,
            volatility: volatilities[index] * 100,
            VaR_95: assetVaR,
            contributionToPortfolioVaR: (assetVaR / VaR) * 100  // Approximate contribution
        };
    });

    // Calculate correlation verification
    const realizedCorrelations = calculateRealizedCorrelations(assetPaths);

    return {
        // Basic GBM results
        ...geometricBrownianMotion(portfolio, params),

        // Multi-asset specific analysis
        assetAnalysis,

        // Correlation Analysis
        correlationAnalysis: {
            inputCorrelations: correlationMatrix,
            realizedCorrelations,
            correlationError: calculateCorrelationError(correlationMatrix, realizedCorrelations),
            diversificationBenefit: calculateDiversificationBenefit(portfolio, assetAnalysis, VaR)
        },

        // Enhanced performance metrics
        performance: {
            simulationTime: simulationTime,
            pathsPerSecond: Math.round(iterations / simulationTime),
            assetsSimulated: numAssets,
            totalComputations: iterations * numAssets
        }
    };
}

/**
 * Path-dependent GBM analysis
 * 
 * Generates full price paths (not just terminal values) for analysis of
 * path-dependent options, drawdowns, and interim risk metrics.
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - Path-dependent simulation parameters
 * @returns {Object} Path-dependent analysis results
 */
function pathDependentGBM(portfolio, params = {}) {
    const {
        iterations = 10000,             // Reduced for path simulation
        confidence = 0.95,
        horizon = 1.0,
        timeSteps = 252,                // Daily time steps
        expectedReturns = [0.08, 0.05, 0.12, 0.03],
        volatilities = [0.20, 0.15, 0.25, 0.10],
        correlationMatrix = [
            [1.00, 0.40, 0.30, 0.10],
            [0.40, 1.00, 0.35, 0.15],
            [0.30, 0.35, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00]
        ]
    } = params;

    console.log(`🛤️  Starting Path-Dependent GBM with ${iterations.toLocaleString()} paths, ${timeSteps} steps...`);
    const startTime = Date.now();

    const numAssets = portfolio.length;
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);
    const weights = portfolio.map(value => value / initialValue);
    const dt = horizon / timeSteps;

    // Arrays to store path-dependent metrics
    const terminalValues = [];
    const maxDrawdowns = [];
    const timeToMaxDrawdown = [];
    const pathVolatilities = [];
    const maxPortfolioValues = [];

    for (let path = 0; path < iterations; path++) {
        // Generate correlated shocks for the entire path
        const pathShocks = generateCorrelatedNormals(timeSteps, correlationMatrix);

        // Initialize asset values
        const assetValues = [...portfolio];
        const portfolioPath = [initialValue];
        let maxPortfolioValue = initialValue;
        let maxDrawdown = 0;
        let timeOfMaxDrawdown = 0;

        // Simulate each time step
        for (let t = 0; t < timeSteps; t++) {
            let portfolioValue = 0;

            // Update each asset
            for (let asset = 0; asset < numAssets; asset++) {
                const drift = (expectedReturns[asset] - 0.5 * volatilities[asset] * volatilities[asset]) * dt;
                const diffusion = volatilities[asset] * Math.sqrt(dt) * pathShocks[t][asset];

                assetValues[asset] *= Math.exp(drift + diffusion);
                portfolioValue += assetValues[asset];
            }

            portfolioPath.push(portfolioValue);

            // Track maximum value and drawdown
            if (portfolioValue > maxPortfolioValue) {
                maxPortfolioValue = portfolioValue;
            }

            const currentDrawdown = (maxPortfolioValue - portfolioValue) / maxPortfolioValue;
            if (currentDrawdown > maxDrawdown) {
                maxDrawdown = currentDrawdown;
                timeOfMaxDrawdown = t + 1;
            }
        }

        // Store path-dependent metrics
        terminalValues.push(portfolioPath[portfolioPath.length - 1]);
        maxDrawdowns.push(maxDrawdown);
        timeToMaxDrawdown.push(timeOfMaxDrawdown);
        maxPortfolioValues.push(maxPortfolioValue);

        // Calculate path volatility (realized volatility)
        const pathReturns = [];
        for (let i = 1; i < portfolioPath.length; i++) {
            pathReturns.push(Math.log(portfolioPath[i] / portfolioPath[i - 1]));
        }
        const pathVol = calculateStatistics(pathReturns).std * Math.sqrt(252);  // Annualized
        pathVolatilities.push(pathVol);
    }

    const simulationTime = (Date.now() - startTime) / 1000;

    // Calculate standard risk metrics
    const losses = terminalValues.map(val => initialValue - val);
    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Calculate path-dependent statistics
    const drawdownStats = calculateStatistics(maxDrawdowns);
    const pathVolStats = calculateStatistics(pathVolatilities);
    const timeToDrawdownStats = calculateStatistics(timeToMaxDrawdown);

    return {
        // Standard simulation results
        simulationConfig: {
            method: 'pathDependentGBM',
            engine: 'node',
            iterations,
            timeSteps,
            horizon,
            confidence,
            initialValue
        },

        // Standard risk metrics
        riskMetrics: {
            VaR_95: VaR,
            CVaR_95: CVaR,
            probabilityOfLoss: (terminalValues.filter(val => val < initialValue).length / iterations) * 100
        },

        // Path-dependent analysis
        pathDependentAnalysis: {
            maxDrawdown: {
                mean: drawdownStats.mean * 100,
                worst: drawdownStats.max * 100,
                p95: calculateVaR(maxDrawdowns.map(dd => -dd), 0.95) * 100,  // 95th percentile worst drawdown
                standardDeviation: drawdownStats.std * 100
            },

            realizedVolatility: {
                mean: pathVolStats.mean * 100,
                standardDeviation: pathVolStats.std * 100,
                min: pathVolStats.min * 100,
                max: pathVolStats.max * 100
            },

            timeToMaxDrawdown: {
                averageDays: timeToDrawdownStats.mean,
                medianDays: timeToDrawdownStats.median,
                range: `${timeToDrawdownStats.min} - ${timeToDrawdownStats.max} days`
            },

            pathEfficiency: {
                averageMaxValue: calculateStatistics(maxPortfolioValues).mean,
                averageTerminalVsMax: (calculateStatistics(terminalValues).mean / calculateStatistics(maxPortfolioValues).mean) * 100
            }
        },

        // Performance metrics
        performance: {
            simulationTime: simulationTime,
            pathsPerSecond: Math.round(iterations / simulationTime),
            totalTimeSteps: iterations * timeSteps
        },

        // Sample path data (first path for visualization)
        samplePath: null  // Could include first simulated path for plotting
    };
}

/**
 * Calculate realized correlations from simulated asset paths
 * 
 * @param {Array<Array<number>>} assetPaths - Paths for each asset
 * @returns {Array<Array<number>>} Realized correlation matrix
 */
function calculateRealizedCorrelations(assetPaths) {
    const numAssets = assetPaths.length;
    const correlationMatrix = Array(numAssets).fill().map(() => Array(numAssets).fill(0));

    // Calculate returns for each asset
    const assetReturns = assetPaths.map(path => {
        const returns = [];
        for (let i = 1; i < path.length; i++) {
            returns.push(Math.log(path[i] / path[i - 1]));
        }
        return returns;
    });

    // Calculate correlation matrix
    for (let i = 0; i < numAssets; i++) {
        for (let j = 0; j < numAssets; j++) {
            if (i === j) {
                correlationMatrix[i][j] = 1.0;
            } else {
                const correlation = calculateCorrelation(assetReturns[i], assetReturns[j]);
                correlationMatrix[i][j] = correlation;
            }
        }
    }

    return correlationMatrix;
}

/**
 * Calculate correlation between two return series
 * 
 * @param {Array<number>} series1 - First return series
 * @param {Array<number>} series2 - Second return series
 * @returns {number} Correlation coefficient
 */
function calculateCorrelation(series1, series2) {
    const n = Math.min(series1.length, series2.length);
    if (n === 0) return 0;

    const mean1 = series1.slice(0, n).reduce((sum, val) => sum + val, 0) / n;
    const mean2 = series2.slice(0, n).reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let sumSquares1 = 0;
    let sumSquares2 = 0;

    for (let i = 0; i < n; i++) {
        const diff1 = series1[i] - mean1;
        const diff2 = series2[i] - mean2;

        numerator += diff1 * diff2;
        sumSquares1 += diff1 * diff1;
        sumSquares2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(sumSquares1 * sumSquares2);
    return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Calculate correlation error between input and realized correlations
 * 
 * @param {Array<Array<number>>} inputCorr - Input correlation matrix
 * @param {Array<Array<number>>} realizedCorr - Realized correlation matrix
 * @returns {Object} Correlation error analysis
 */
function calculateCorrelationError(inputCorr, realizedCorr) {
    const errors = [];

    for (let i = 0; i < inputCorr.length; i++) {
        for (let j = i + 1; j < inputCorr[i].length; j++) {
            const error = Math.abs(inputCorr[i][j] - realizedCorr[i][j]);
            errors.push(error);
        }
    }

    const errorStats = calculateStatistics(errors);

    return {
        meanAbsoluteError: errorStats.mean,
        maxError: errorStats.max,
        rmsError: Math.sqrt(errors.reduce((sum, err) => sum + err * err, 0) / errors.length)
    };
}

/**
 * Calculate diversification benefit from multi-asset analysis
 * 
 * @param {Array<number>} portfolio - Portfolio values
 * @param {Array<Object>} assetAnalysis - Asset-level analysis
 * @param {number} portfolioVaR - Portfolio VaR
 * @returns {Object} Diversification analysis
 */
function calculateDiversificationBenefit(portfolio, assetAnalysis, portfolioVaR) {
    const initialValue = portfolio.reduce((sum, val) => sum + val, 0);

    // Calculate standalone VaR for each asset
    const standaloneVaRs = assetAnalysis.map(asset => asset.VaR_95);
    const weightedStandaloneVaR = standaloneVaRs.reduce((sum, var_, idx) =>
        sum + var_ * (portfolio[idx] / initialValue), 0);

    // Diversification ratio
    const diversificationRatio = portfolioVaR / weightedStandaloneVaR;
    const diversificationBenefit = weightedStandaloneVaR - portfolioVaR;

    return {
        portfolioVaR,
        weightedStandaloneVaR,
        diversificationBenefit,
        diversificationRatio,
        benefitPercentage: (diversificationBenefit / weightedStandaloneVaR) * 100
    };
}

module.exports = {
    geometricBrownianMotion,
    multiAssetGBM,
    pathDependentGBM
};
