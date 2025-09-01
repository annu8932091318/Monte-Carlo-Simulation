/**
 * Historical Simulation for Portfolio Risk Assessment
 * 
 * Historical simulation uses actual historical returns to model portfolio risk.
 * It assumes that the future distribution of returns will be similar to the past.
 * This method doesn't assume any particular distribution (non-parametric).
 */

const {
    calculateVaR,
    calculateCVaR,
    calculateStatistics,
    bootstrapSample
} = require('../utils/mathHelpers');

/**
 * Generate sample historical returns data
 * In a real implementation, this would come from a financial data API
 * 
 * @param {number} days - Number of historical days to generate
 * @returns {Array<Array<number>>} Array of daily returns for each asset
 */
function generateSampleHistoricalReturns(days = 252) {
    // Simulate 1 year of daily returns for 4 assets
    const assets = 4;
    const returns = Array(assets).fill().map(() => []);

    // Base parameters for realistic-looking historical data
    const meanReturns = [0.08 / 252, 0.05 / 252, 0.12 / 252, 0.03 / 252];  // Daily expected returns
    const volatilities = [0.20 / Math.sqrt(252), 0.15 / Math.sqrt(252), 0.25 / Math.sqrt(252), 0.10 / Math.sqrt(252)];

    for (let day = 0; day < days; day++) {
        for (let asset = 0; asset < assets; asset++) {
            // Add some autocorrelation and volatility clustering for realism
            const prevReturn = returns[asset][day - 1] || 0;
            const volatilityAdjustment = 1 + 0.3 * Math.abs(prevReturn);

            let u1 = 0, u2 = 0;
            while (u1 === 0) u1 = Math.random();
            while (u2 === 0) u2 = Math.random();

            const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            const dailyReturn = meanReturns[asset] + volatilities[asset] * volatilityAdjustment * z;

            returns[asset].push(dailyReturn);
        }
    }

    return returns;
}

/**
 * Historical simulation for portfolio risk
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - Simulation parameters
 * @returns {Object} Risk metrics based on historical simulation
 */
function historicalSimulation(portfolio, params = {}) {
    const {
        confidence = 0.95,
        horizon = 1,                    // Horizon in days
        lookbackPeriod = 252,           // Historical lookback period (1 year)
        historicalReturns = null        // Optional: provide your own historical data
    } = params;

    console.log(`📊 Starting Historical Simulation with ${lookbackPeriod} day lookback...`);
    const startTime = Date.now();

    // Get historical returns (either provided or generated)
    const returns = historicalReturns || generateSampleHistoricalReturns(lookbackPeriod);
    const numAssets = portfolio.length;
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);
    const weights = portfolio.map(value => value / initialValue);

    // Calculate portfolio returns for each historical period
    const portfolioReturns = [];
    const numPeriods = returns[0].length;

    for (let day = 0; day < numPeriods; day++) {
        let portfolioReturn = 0;
        for (let asset = 0; asset < numAssets; asset++) {
            portfolioReturn += weights[asset] * returns[asset][day];
        }
        portfolioReturns.push(portfolioReturn);
    }

    // Scale returns to the desired horizon
    const scaledReturns = portfolioReturns.map(ret => ret * horizon);

    // Calculate potential portfolio values
    const terminalValues = scaledReturns.map(ret => initialValue * (1 + ret));
    const losses = terminalValues.map(val => initialValue - val);

    const simulationTime = (Date.now() - startTime) / 1000;

    // Calculate risk metrics
    const VaR = calculateVaR(losses, confidence);
    const CVaR = calculateCVaR(losses, confidence);

    // Calculate statistics
    const terminalStats = calculateStatistics(terminalValues);
    const returnStats = calculateStatistics(scaledReturns);

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
            method: 'historicalSimulation',
            engine: 'node',
            lookbackPeriod,
            horizon,
            confidence,
            initialValue,
            scenarios: numPeriods
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
            minReturn: returnStats.min * 100,
            maxReturn: returnStats.max * 100
        },

        // Scenario Analysis
        scenarios: {
            pessimistic_p10: p10,
            median_p50: p50,
            optimistic_p90: p90
        },

        // Historical Analysis
        historicalAnalysis: {
            worstHistoricalReturn: Math.min(...scaledReturns) * 100,
            bestHistoricalReturn: Math.max(...scaledReturns) * 100,
            averageReturn: returnStats.mean * 100,
            historicalVolatility: returnStats.std * 100,
            negativeReturnDays: scaledReturns.filter(ret => ret < 0).length,
            negativeReturnPercentage: (scaledReturns.filter(ret => ret < 0).length / scaledReturns.length) * 100
        },

        // Performance Metrics
        performance: {
            simulationTime: simulationTime,
            scenariosProcessed: numPeriods
        },

        // Raw data for plotting (limited to avoid large payloads)
        distribution: terminalValues
    };
}

/**
 * Bootstrap Historical Simulation
 * 
 * Enhances historical simulation by bootstrap resampling to increase
 * the number of scenarios and smooth out small sample biases.
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - Bootstrap simulation parameters
 * @returns {Object} Risk metrics with bootstrap enhancement
 */
function bootstrapHistoricalSimulation(portfolio, params = {}) {
    const {
        confidence = 0.95,
        horizon = 1,
        lookbackPeriod = 252,
        bootstrapSamples = 10000,       // Number of bootstrap samples
        historicalReturns = null
    } = params;

    console.log(`🔄 Starting Bootstrap Historical Simulation with ${bootstrapSamples.toLocaleString()} samples...`);
    const startTime = Date.now();

    // Get historical returns
    const returns = historicalReturns || generateSampleHistoricalReturns(lookbackPeriod);
    const numAssets = portfolio.length;
    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);
    const weights = portfolio.map(value => value / initialValue);

    // Calculate historical portfolio returns
    const portfolioReturns = [];
    const numPeriods = returns[0].length;

    for (let day = 0; day < numPeriods; day++) {
        let portfolioReturn = 0;
        for (let asset = 0; asset < numAssets; asset++) {
            portfolioReturn += weights[asset] * returns[asset][day];
        }
        portfolioReturns.push(portfolioReturn);
    }

    // Generate bootstrap samples
    const bootstrapReturns = [];
    for (let i = 0; i < bootstrapSamples; i++) {
        const sample = bootstrapSample(portfolioReturns, 1)[0];  // Sample single return
        bootstrapReturns.push(sample * horizon);
    }

    // Calculate potential portfolio values from bootstrap samples
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

    // Get original historical simulation for comparison
    const originalHistSim = historicalSimulation(portfolio, {
        ...params,
        historicalReturns: returns
    });

    return {
        // Enhanced simulation config
        simulationConfig: {
            method: 'bootstrapHistoricalSimulation',
            engine: 'node',
            lookbackPeriod,
            horizon,
            confidence,
            initialValue,
            bootstrapSamples,
            originalScenarios: numPeriods
        },

        // Risk Metrics (from bootstrap)
        riskMetrics: {
            VaR_95: VaR,
            CVaR_95: CVaR,
            probabilityOfLoss: probabilityOfLoss * 100,
            expectedShortfall: CVaR
        },

        // Comparison with original historical simulation
        comparison: {
            bootstrap: {
                VaR_95: VaR,
                CVaR_95: CVaR,
                expectedReturn: returnStats.mean * 100,
                volatility: returnStats.std * 100
            },
            original: {
                VaR_95: originalHistSim.riskMetrics.VaR_95,
                CVaR_95: originalHistSim.riskMetrics.CVaR_95,
                expectedReturn: originalHistSim.portfolioStats.expectedReturn,
                volatility: originalHistSim.portfolioStats.volatility
            }
        },

        // Portfolio Statistics
        portfolioStats: {
            expectedTerminal: terminalStats.mean,
            standardDeviation: terminalStats.std,
            expectedReturn: returnStats.mean * 100,
            volatility: returnStats.std * 100
        },

        // Performance Metrics
        performance: {
            simulationTime: simulationTime,
            samplesPerSecond: Math.round(bootstrapSamples / simulationTime)
        },

        // Distribution (subset for plotting)
        distribution: terminalValues.slice(0, Math.min(10000, bootstrapSamples))
    };
}

module.exports = {
    historicalSimulation,
    bootstrapHistoricalSimulation,
    generateSampleHistoricalReturns
};
