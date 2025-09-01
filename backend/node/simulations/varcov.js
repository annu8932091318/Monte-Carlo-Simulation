/**
 * Variance-Covariance (Parametric) Method for Risk Assessment
 * 
 * The VarCov method assumes that portfolio returns follow a normal distribution
 * and calculates VaR analytically using the portfolio's mean and standard deviation.
 * This is the fastest method but relies on normality assumptions.
 */

const {
    calculatePortfolioStats,
    correlationToCovariance,
    calculateStatistics
} = require('../utils/mathHelpers');

/**
 * Variance-Covariance VaR calculation
 * 
 * Analytical VaR calculation assuming normal distribution:
 * VaR = μ - z_α * σ * √T
 * 
 * Where:
 * - μ is the expected portfolio return
 * - σ is the portfolio volatility  
 * - z_α is the z-score for confidence level α
 * - T is the time horizon
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} params - VarCov parameters
 * @returns {Object} Analytical risk metrics
 */
function varianceCovarianceVaR(portfolio, params = {}) {
    const {
        confidence = 0.95,
        horizon = 1.0,                  // Time horizon in years
        expectedReturns = [0.08, 0.05, 0.12, 0.03],  // Annual expected returns
        volatilities = [0.20, 0.15, 0.25, 0.10],     // Annual volatilities
        correlationMatrix = [
            [1.00, 0.40, 0.30, 0.10],
            [0.40, 1.00, 0.35, 0.15],
            [0.30, 0.35, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00]
        ]
    } = params;

    console.log(`🧮 Starting Variance-Covariance VaR calculation...`);
    const startTime = Date.now();

    const initialValue = portfolio.reduce((sum, value) => sum + value, 0);
    const weights = portfolio.map(value => value / initialValue);

    // Convert correlation to covariance matrix
    const covarianceMatrix = correlationToCovariance(volatilities, correlationMatrix);

    // Calculate portfolio statistics
    const portfolioStats = calculatePortfolioStats(weights, expectedReturns, covarianceMatrix);
    const { expectedReturn: mu_p, volatility: sigma_p } = portfolioStats;

    // Z-scores for different confidence levels
    const zScores = {
        0.90: 1.282,   // 90% confidence
        0.95: 1.645,   // 95% confidence  
        0.99: 2.326,   // 99% confidence
        0.999: 3.090   // 99.9% confidence
    };

    const zScore = zScores[confidence] || 1.645;  // Default to 95%

    // Scale parameters for the time horizon
    const horizonMu = mu_p * horizon;
    const horizonSigma = sigma_p * Math.sqrt(horizon);

    // Calculate analytical VaR and CVaR
    // VaR = Expected Loss = -(μ - z_α * σ) for the worst case
    const expectedPortfolioValue = initialValue * (1 + horizonMu);
    const portfolioStdDev = initialValue * horizonSigma;

    // VaR: potential loss at confidence level
    const VaR = -(horizonMu * initialValue - zScore * portfolioStdDev);

    // CVaR for normal distribution: CVaR = VaR + σ * φ(z_α) / (1-α)
    // Where φ is the standard normal PDF
    const phi = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * zScore * zScore);
    const CVaR = VaR + portfolioStdDev * phi / (1 - confidence);

    // Calculate probability of loss (using cumulative normal distribution)
    // P(Loss) = P(Return < 0) = Φ(-μ/σ)
    const zLoss = -horizonMu / horizonSigma;
    const probabilityOfLoss = normalCDF(zLoss) * 100;

    // Calculate percentiles using normal distribution
    const p10Value = initialValue * (1 + horizonMu + normalInverseCDF(0.10) * horizonSigma);
    const p50Value = initialValue * (1 + horizonMu);  // Median = mean for normal dist
    const p90Value = initialValue * (1 + horizonMu + normalInverseCDF(0.90) * horizonSigma);

    const calculationTime = (Date.now() - startTime) / 1000;

    return {
        // Simulation Configuration
        simulationConfig: {
            method: 'varianceCovariance',
            engine: 'node',
            horizon,
            confidence,
            initialValue,
            portfolioMu: mu_p,
            portfolioSigma: sigma_p,
            assumptionNote: 'Assumes normal distribution of returns'
        },

        // Risk Metrics (Analytical)
        riskMetrics: {
            VaR_95: VaR,
            CVaR_95: CVaR,
            probabilityOfLoss: probabilityOfLoss,
            expectedShortfall: CVaR
        },

        // Portfolio Statistics
        portfolioStats: {
            expectedTerminal: expectedPortfolioValue,
            standardDeviation: portfolioStdDev,
            expectedReturn: horizonMu * 100,
            volatility: sigma_p * 100,
            sharpeRatio: (mu_p - 0.02) / sigma_p  // Assuming 2% risk-free rate
        },

        // Scenario Analysis (based on normal distribution)
        scenarios: {
            pessimistic_p10: p10Value,
            median_p50: p50Value,
            optimistic_p90: p90Value
        },

        // Additional VaR metrics for different confidence levels
        varLevels: {
            VaR_90: -(horizonMu * initialValue - zScores[0.90] * portfolioStdDev),
            VaR_95: VaR,
            VaR_99: -(horizonMu * initialValue - zScores[0.99] * portfolioStdDev),
            VaR_999: -(horizonMu * initialValue - zScores[0.999] * portfolioStdDev)
        },

        // Component Analysis (contribution of each asset to portfolio risk)
        componentAnalysis: calculateComponentVaR(portfolio, weights, expectedReturns,
            covarianceMatrix, confidence, horizon),

        // Performance Metrics
        performance: {
            calculationTime: calculationTime,
            method: 'analytical'
        },

        // Model Assumptions
        assumptions: {
            distribution: 'normal',
            independence: false,
            correlationModeled: true,
            timePeriod: `${horizon} year(s)`,
            limitationsNote: 'May underestimate tail risks and fat-tail events'
        }
    };
}

/**
 * Calculate Component VaR - contribution of each asset to total portfolio VaR
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Array<number>} weights - Portfolio weights
 * @param {Array<number>} expectedReturns - Expected returns
 * @param {Array<Array<number>>} covarianceMatrix - Covariance matrix
 * @param {number} confidence - Confidence level
 * @param {number} horizon - Time horizon
 * @returns {Object} Component VaR analysis
 */
function calculateComponentVaR(portfolio, weights, expectedReturns, covarianceMatrix, confidence, horizon) {
    const zScore = confidence === 0.95 ? 1.645 : 1.282;  // Simplified
    const initialValue = portfolio.reduce((sum, val) => sum + val, 0);

    // Calculate marginal VaR for each asset
    const portfolioStats = calculatePortfolioStats(weights, expectedReturns, covarianceMatrix);
    const portfolioVolatility = portfolioStats.volatility;

    const componentVaRs = [];
    const marginalVaRs = [];

    for (let i = 0; i < weights.length; i++) {
        // Marginal VaR: ∂VaR/∂w_i = z_α * (Σw)_i / σ_p
        let marginalContribution = 0;
        for (let j = 0; j < weights.length; j++) {
            marginalContribution += weights[j] * covarianceMatrix[i][j];
        }

        const marginalVaR = zScore * Math.sqrt(horizon) * marginalContribution / portfolioVolatility;
        marginalVaRs.push(marginalVaR);

        // Component VaR: w_i * MarginalVaR_i
        const componentVaR = weights[i] * marginalVaR * initialValue;
        componentVaRs.push(componentVaR);
    }

    return {
        componentVaRs: componentVaRs.map((cvar, idx) => ({
            assetIndex: idx,
            assetValue: portfolio[idx],
            weight: weights[idx],
            componentVaR: cvar,
            percentageContribution: (cvar / componentVaRs.reduce((a, b) => a + b, 0)) * 100
        })),
        marginalVaRs,
        totalComponentVaR: componentVaRs.reduce((a, b) => a + b, 0)
    };
}

/**
 * Standard normal cumulative distribution function (approximation)
 * 
 * @param {number} z - Z-score
 * @returns {number} Cumulative probability
 */
function normalCDF(z) {
    // Abramowitz and Stegun approximation
    const t = 1.0 / (1.0 + 0.2316419 * Math.abs(z));
    const d = 0.3989423 * Math.exp(-z * z / 2);
    let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if (z > 0) {
        prob = 1 - prob;
    }

    return prob;
}

/**
 * Inverse normal cumulative distribution function (approximation)
 * 
 * @param {number} p - Probability (0 < p < 1)
 * @returns {number} Z-score
 */
function normalInverseCDF(p) {
    // Beasley-Springer-Moro algorithm
    if (p <= 0 || p >= 1) {
        throw new Error('Probability must be between 0 and 1');
    }

    const c = [2.515517, 0.802853, 0.010328];
    const d = [1.432788, 0.189269, 0.001308];

    if (p < 0.5) {
        const t = Math.sqrt(-2 * Math.log(p));
        return -((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1);
    } else {
        const t = Math.sqrt(-2 * Math.log(1 - p));
        return ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1);
    }
}

/**
 * Stress testing using VarCov method
 * 
 * @param {Array<number>} portfolio - Portfolio asset values
 * @param {Object} stressParams - Stress test parameters
 * @returns {Object} Stress test results
 */
function stressTestVarCov(portfolio, stressParams = {}) {
    const {
        baseParams = {},
        stressScenarios = [
            { name: 'Market Crash', returnShock: -0.20, volShock: 2.0, corrShock: 1.5 },
            { name: 'High Volatility', returnShock: 0, volShock: 1.5, corrShock: 1.2 },
            { name: 'Low Returns', returnShock: -0.10, volShock: 1.0, corrShock: 1.0 }
        ]
    } = stressParams;

    console.log(`🚨 Running VarCov Stress Tests...`);

    // Base case
    const baseCase = varianceCovarianceVaR(portfolio, baseParams);

    // Stress scenarios
    const stressResults = stressScenarios.map(scenario => {
        const stressedParams = { ...baseParams };

        // Apply shocks
        if (stressedParams.expectedReturns) {
            stressedParams.expectedReturns = stressedParams.expectedReturns.map(ret =>
                ret + scenario.returnShock);
        }

        if (stressedParams.volatilities) {
            stressedParams.volatilities = stressedParams.volatilities.map(vol =>
                vol * scenario.volShock);
        }

        if (stressedParams.correlationMatrix && scenario.corrShock !== 1.0) {
            stressedParams.correlationMatrix = stressedParams.correlationMatrix.map(row =>
                row.map((corr, idx) => {
                    if (idx === row.indexOf(corr) && corr === 1.0) return 1.0;  // Keep diagonal at 1
                    return Math.min(0.99, corr * scenario.corrShock);  // Cap at 0.99
                })
            );
        }

        const stressResult = varianceCovarianceVaR(portfolio, stressedParams);

        return {
            scenarioName: scenario.name,
            shocks: scenario,
            results: stressResult.riskMetrics,
            comparison: {
                varIncrease: stressResult.riskMetrics.VaR_95 - baseCase.riskMetrics.VaR_95,
                varMultiplier: stressResult.riskMetrics.VaR_95 / baseCase.riskMetrics.VaR_95
            }
        };
    });

    return {
        baseCase: baseCase.riskMetrics,
        stressResults,
        summary: {
            worstCaseVaR: Math.max(...stressResults.map(s => s.results.VaR_95)),
            averageStressVaR: stressResults.reduce((sum, s) => sum + s.results.VaR_95, 0) / stressResults.length
        }
    };
}

module.exports = {
    varianceCovarianceVaR,
    stressTestVarCov,
    normalCDF,
    normalInverseCDF
};
