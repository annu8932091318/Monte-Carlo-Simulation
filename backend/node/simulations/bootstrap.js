/**
 * Bootstrap Simulation for Portfolio Risk Assessment
 * 
 * Bootstrap method uses resampling with replacement from historical data
 * to generate new scenarios. This non-parametric approach doesn't assume
 * any particular distribution and can capture complex patterns in the data.
 */

const {
    calculateVaR,
    calculateCVaR,
    calculateStatistics,
    bootstrapSample
} = require('../utils/mathHelpers');

/**
 * Generate sample time series data for bootstrap simulation
 * In practice, this would be replaced with real historical price/return data
 * 
 * @param {number} periods - Number of time periods
 * @param {number} assets - Number of assets
 * @returns {Object} Historical prices and returns data
 */
function generateSampleTimeSeriesData(periods = 252, assets = 4) {
    // Generate realistic-looking price paths with different characteristics
    const initialPrices = [100, 50, 75, 200];  // Starting prices for each asset
    const prices = Array(assets).fill().map((_, i) => [initialPrices[i]]);
    const returns = Array(assets).fill().map(() => []);

    // Different asset characteristics
    const assetParams = [
        { mu: 0.08 / 252, sigma: 0.20 / Math.sqrt(252), trending: true },   // Growth stock
        { mu: 0.05 / 252, sigma: 0.15 / Math.sqrt(252), trending: false },  // Stable bond
        { mu: 0.12 / 252, sigma: 0.25 / Math.sqrt(252), trending: true },   // High-growth stock
        { mu: 0.03 / 252, sigma: 0.10 / Math.sqrt(252), trending: false }   // Cash-like
    ];

    for (let t = 1; t <= periods; t++) {
        for (let asset = 0; asset < assets; asset++) {
            const { mu, sigma, trending } = assetParams[asset];
            const prevPrice = prices[asset][t - 1];

            // Add momentum and mean reversion effects
            const prevReturn = returns[asset][t - 2] || 0;
            const momentum = trending ? 0.1 * prevReturn : -0.05 * prevReturn;  // Mean reversion for bonds

            // Generate random shock
            let u1 = 0, u2 = 0;
            while (u1 === 0) u1 = Math.random();
            while (u2 === 0) u2 = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

            // Calculate return with momentum
            const dailyReturn = mu + momentum + sigma * z;
            returns[asset].push(dailyReturn);

            // Calculate new price
            const newPrice = prevPrice * (1 + dailyReturn);
            prices[asset].push(newPrice);
        }
    }

    return { prices, returns };
}

/**
 * Simple Bootstrap simulation
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - Bootstrap parameters
 * @returns {Object} Bootstrap simulation results
 */
function bootstrapSimulation(portfolio, params = {}) {
    const {
        confidence = 0.95,
        horizon = 1,                    // Horizon in periods
        bootstrapSamples = 10000,       // Number of bootstrap samples
        blockSize = 1,                  // Block size for block bootstrap
        historicalData = null           // Optional historical data
    } = params;

    console.log(`🔄 Starting Bootstrap Simulation with ${bootstrapSamples.toLocaleString()} samples...`);
    const startTime = Date.now();

    // Get or generate historical data
    const data = historicalData || generateSampleTimeSeriesData();
    const { returns } = data;

    const numAssets = portfolio.length;
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);
    const weights = portfolio.map(value => value / initialValue);

    // Calculate historical portfolio returns
    const historicalPortfolioReturns = [];
    const numPeriods = returns[0].length;

    for (let t = 0; t < numPeriods; t++) {
        let portfolioReturn = 0;
        for (let asset = 0; asset < numAssets; asset++) {
            portfolioReturn += weights[asset] * returns[asset][t];
        }
        historicalPortfolioReturns.push(portfolioReturn);
    }

    // Generate bootstrap samples
    const bootstrapReturns = [];
    for (let sample = 0; sample < bootstrapSamples; sample++) {
        if (blockSize === 1) {
            // Simple bootstrap: sample single returns
            const sampledReturns = bootstrapSample(historicalPortfolioReturns, horizon);
            const cumulativeReturn = sampledReturns.reduce((cum, ret) => cum * (1 + ret), 1) - 1;
            bootstrapReturns.push(cumulativeReturn);
        } else {
            // Block bootstrap: sample blocks of consecutive returns
            const blocks = [];
            for (let i = 0; i <= numPeriods - blockSize; i++) {
                blocks.push(historicalPortfolioReturns.slice(i, i + blockSize));
            }

            const numBlocksNeeded = Math.ceil(horizon / blockSize);
            let cumulativeReturn = 0;

            for (let block = 0; block < numBlocksNeeded; block++) {
                const selectedBlock = bootstrapSample(blocks, 1)[0];
                const returnsToUse = selectedBlock.slice(0, Math.min(blockSize, horizon - block * blockSize));

                for (const ret of returnsToUse) {
                    cumulativeReturn = (1 + cumulativeReturn) * (1 + ret) - 1;
                }
            }

            bootstrapReturns.push(cumulativeReturn);
        }
    }

    // Calculate terminal portfolio values
    const terminalValues = bootstrapReturns.map(ret => initialValue * (1 + ret));
    const losses = terminalValues.map(val => initialValue - val);

    const simulationTime = (Date.now() - startTime) / 1000;

    // Calculate risk metrics
    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Calculate statistics
    const terminalStats = calculateStatistics(terminalValues);
    const returnStats = calculateStatistics(bootstrapReturns);

    // Calculate probability of loss
    const lossCount = terminalValues.filter(val => val < initialValue).length;
    const probabilityOfLoss = lossCount / terminalValues.length;

    // Calculate percentiles
    const sortedTerminals = [...terminalValues].sort((a, b) => a - b);
    const p10 = sortedTerminals[Math.floor(0.10 * sortedTerminals.length)];
    const p50 = sortedTerminals[Math.floor(0.50 * sortedTerminals.length)];
    const p90 = sortedTerminals[Math.floor(0.90 * sortedTerminals.length)];

    return {
        // Simulation Configuration
        simulationConfig: {
            method: 'bootstrap',
            engine: 'node',
            bootstrapSamples,
            blockSize,
            horizon,
            confidence,
            initialValue,
            historicalPeriods: numPeriods
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
            expectedReturn: returnStats.mean * 100,
            volatility: returnStats.std * 100,
            skewness: calculateSkewness(bootstrapReturns),
            kurtosis: calculateKurtosis(bootstrapReturns)
        },

        // Scenario Analysis
        scenarios: {
            pessimistic_p10: p10,
            median_p50: p50,
            optimistic_p90: p90
        },

        // Bootstrap-specific analysis
        bootstrapAnalysis: {
            originalSampleSize: numPeriods,
            bootstrapSampleSize: bootstrapSamples,
            samplingMethod: blockSize === 1 ? 'simple' : `block (size ${blockSize})`,
            distributionShape: {
                skewness: calculateSkewness(bootstrapReturns),
                kurtosis: calculateKurtosis(bootstrapReturns),
                isNormalLike: Math.abs(calculateSkewness(bootstrapReturns)) < 0.5 &&
                    Math.abs(calculateKurtosis(bootstrapReturns) - 3) < 1
            }
        },

        // Performance Metrics
        performance: {
            simulationTime: simulationTime,
            samplesPerSecond: Math.round(bootstrapSamples / simulationTime)
        },

        // Distribution data (subset for plotting)
        distribution: terminalValues.slice(0, Math.min(10000, bootstrapSamples))
    };
}

/**
 * Advanced Block Bootstrap simulation
 * 
 * Uses overlapping block bootstrap to better preserve time series dependencies
 * and autocorrelation patterns in the data.
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - Advanced bootstrap parameters
 * @returns {Object} Advanced bootstrap results
 */
function advancedBlockBootstrap(portfolio, params = {}) {
    const {
        confidence = 0.95,
        horizon = 22,                   // ~1 month in trading days
        bootstrapSamples = 10000,
        blockSize = 5,                  // 1 week blocks
        overlapRatio = 0.5,             // 50% overlap between blocks
        historicalData = null
    } = params;

    console.log(`🧱 Starting Advanced Block Bootstrap with ${bootstrapSamples.toLocaleString()} samples...`);
    const startTime = Date.now();

    // Get or generate historical data
    const data = historicalData || generateSampleTimeSeriesData(500);  // Longer history for blocks
    const { returns } = data;

    const numAssets = portfolio.length;
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);
    const weights = portfolio.map(value => value / initialValue);

    // Calculate historical portfolio returns
    const historicalPortfolioReturns = [];
    const numPeriods = returns[0].length;

    for (let t = 0; t < numPeriods; t++) {
        let portfolioReturn = 0;
        for (let asset = 0; asset < numAssets; asset++) {
            portfolioReturn += weights[asset] * returns[asset][t];
        }
        historicalPortfolioReturns.push(portfolioReturn);
    }

    // Create overlapping blocks
    const stepSize = Math.max(1, Math.floor(blockSize * (1 - overlapRatio)));
    const blocks = [];

    for (let i = 0; i <= numPeriods - blockSize; i += stepSize) {
        blocks.push(historicalPortfolioReturns.slice(i, i + blockSize));
    }

    console.log(`📦 Created ${blocks.length} overlapping blocks of size ${blockSize}`);

    // Generate bootstrap paths
    const bootstrapReturns = [];

    for (let sample = 0; sample < bootstrapSamples; sample++) {
        const path = [];
        let remainingPeriods = horizon;

        while (remainingPeriods > 0) {
            // Select random block
            const selectedBlock = blocks[Math.floor(Math.random() * blocks.length)];
            const periodsToTake = Math.min(remainingPeriods, selectedBlock.length);

            // Take returns from the block
            for (let i = 0; i < periodsToTake; i++) {
                path.push(selectedBlock[i]);
            }

            remainingPeriods -= periodsToTake;
        }

        // Calculate cumulative return for the path
        const cumulativeReturn = path.reduce((cum, ret) => cum * (1 + ret), 1) - 1;
        bootstrapReturns.push(cumulativeReturn);
    }

    // Calculate terminal values and risk metrics (same as simple bootstrap)
    const terminalValues = bootstrapReturns.map(ret => initialValue * (1 + ret));
    const losses = terminalValues.map(val => initialValue - val);

    const simulationTime = (Date.now() - startTime) / 1000;

    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Enhanced analysis with autocorrelation
    const autocorrelation = calculateAutocorrelation(historicalPortfolioReturns, 5);

    return {
        // Enhanced simulation config
        simulationConfig: {
            method: 'advancedBlockBootstrap',
            engine: 'node',
            bootstrapSamples,
            blockSize,
            overlapRatio,
            horizon,
            confidence,
            initialValue,
            numBlocks: blocks.length
        },

        // Risk Metrics
        riskMetrics: {
            VaR_95: VaR,
            CVaR_95: CVaR,
            probabilityOfLoss: (terminalValues.filter(val => val < initialValue).length / terminalValues.length) * 100,
            expectedShortfall: CVaR
        },

        // Time Series Analysis
        timeSeriesAnalysis: {
            autocorrelationLags: autocorrelation,
            persistencePreserved: autocorrelation[0] > 0.1,  // Check if first-order correlation preserved
            blockEffectiveness: {
                description: 'How well the block structure preserves time dependencies',
                score: autocorrelation.reduce((sum, corr) => sum + Math.abs(corr), 0) / autocorrelation.length
            }
        },

        // Performance
        performance: {
            simulationTime: simulationTime,
            samplesPerSecond: Math.round(bootstrapSamples / simulationTime),
            blocksGenerated: blocks.length
        },

        // Distribution (subset)
        distribution: terminalValues.slice(0, Math.min(10000, bootstrapSamples))
    };
}

/**
 * Calculate skewness of a data series
 * 
 * @param {Array<number>} data - Data array
 * @returns {number} Skewness value
 */
function calculateSkewness(data) {
    const n = data.length;
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);

    if (std === 0) return 0;

    const skewness = data.reduce((sum, val) => sum + Math.pow((val - mean) / std, 3), 0) / n;
    return skewness;
}

/**
 * Calculate kurtosis of a data series
 * 
 * @param {Array<number>} data - Data array  
 * @returns {number} Kurtosis value
 */
function calculateKurtosis(data) {
    const n = data.length;
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);

    if (std === 0) return 0;

    const kurtosis = data.reduce((sum, val) => sum + Math.pow((val - mean) / std, 4), 0) / n;
    return kurtosis;
}

/**
 * Calculate autocorrelation for different lags
 * 
 * @param {Array<number>} data - Time series data
 * @param {number} maxLag - Maximum lag to calculate
 * @returns {Array<number>} Autocorrelation values
 */
function calculateAutocorrelation(data, maxLag = 5) {
    const n = data.length;
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;

    const autocorrelations = [];

    for (let lag = 1; lag <= maxLag; lag++) {
        let covariance = 0;
        const count = n - lag;

        for (let i = 0; i < count; i++) {
            covariance += (data[i] - mean) * (data[i + lag] - mean);
        }

        covariance /= count;
        const correlation = variance > 0 ? covariance / variance : 0;
        autocorrelations.push(correlation);
    }

    return autocorrelations;
}

module.exports = {
    bootstrapSimulation,
    advancedBlockBootstrap,
    generateSampleTimeSeriesData
};
