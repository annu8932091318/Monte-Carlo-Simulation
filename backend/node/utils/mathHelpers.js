/**
 * Mathematical Helper Functions for Financial Risk Simulations
 * 
 * This module provides core mathematical utilities used across different
 * simulation strategies including random number generation, statistical
 * functions, and financial calculations.
 */

const math = require('mathjs');

/**
 * Generate random number from normal distribution using Box-Muller transform
 * 
 * @param {number} mean - Mean of the distribution (default: 0)
 * @param {number} std - Standard deviation (default: 1)
 * @returns {number} Random number from normal distribution
 */
function randomNormal(mean = 0, std = 1) {
    // Box-Muller transformation for generating normal random variables
    let u1 = 0, u2 = 0;
    // Ensure we don't get log(0)
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();

    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + std * randStdNormal;
}

/**
 * Generate array of random normal numbers
 * 
 * @param {number} size - Number of random numbers to generate
 * @param {number} mean - Mean of the distribution
 * @param {number} std - Standard deviation
 * @returns {Array<number>} Array of random normal numbers
 */
function randomNormalArray(size, mean = 0, std = 1) {
    const result = [];
    for (let i = 0; i < size; i++) {
        result.push(randomNormal(mean, std));
    }
    return result;
}

/**
 * Calculate percentile of an array
 * 
 * @param {Array<number>} arr - Sorted array of numbers
 * @param {number} percentile - Percentile to calculate (0-100)
 * @returns {number} Value at the specified percentile
 */
function calculatePercentile(arr, percentile) {
    if (arr.length === 0) return 0;

    const sorted = [...arr].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);

    if (Math.floor(index) === index) {
        return sorted[index];
    } else {
        const lower = sorted[Math.floor(index)];
        const upper = sorted[Math.ceil(index)];
        return lower + (upper - lower) * (index - Math.floor(index));
    }
}

/**
 * Calculate Value at Risk (VaR) from a distribution of losses
 * 
 * @param {Array<number>} losses - Array of loss values
 * @param {number} confidence - Confidence level (e.g., 0.95 for 95%)
 * @returns {number} VaR value
 */
function calculateVaR(losses, confidence = 0.95) {
    const percentileValue = confidence * 100;
    return calculatePercentile(losses, percentileValue);
}

/**
 * Calculate Conditional Value at Risk (CVaR) - Expected Shortfall
 * 
 * @param {Array<number>} losses - Array of loss values
 * @param {number} confidence - Confidence level (e.g., 0.95 for 95%)
 * @returns {number} CVaR value
 */
function calculateCVaR(losses, confidence = 0.95) {
    const sorted = [...losses].sort((a, b) => a - b);
    const varIndex = Math.floor((1 - confidence) * sorted.length);

    // CVaR is the average of losses beyond VaR
    if (varIndex === 0) return sorted[0];

    const tailLosses = sorted.slice(0, varIndex);
    return tailLosses.reduce((sum, loss) => sum + loss, 0) / tailLosses.length;
}

/**
 * Calculate portfolio statistics (mean, variance, volatility)
 * 
 * @param {Array<number>} weights - Portfolio weights (must sum to 1)
 * @param {Array<number>} expectedReturns - Expected returns for each asset
 * @param {Array<Array<number>>} covarianceMatrix - Covariance matrix
 * @returns {Object} Portfolio statistics
 */
function calculatePortfolioStats(weights, expectedReturns, covarianceMatrix) {
    // Portfolio expected return: w^T * μ
    const portfolioReturn = weights.reduce((sum, weight, i) =>
        sum + weight * expectedReturns[i], 0);

    // Portfolio variance: w^T * Σ * w
    let portfolioVariance = 0;
    for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights.length; j++) {
            portfolioVariance += weights[i] * weights[j] * covarianceMatrix[i][j];
        }
    }

    const portfolioVolatility = Math.sqrt(portfolioVariance);

    return {
        expectedReturn: portfolioReturn,
        variance: portfolioVariance,
        volatility: portfolioVolatility
    };
}

/**
 * Create correlation matrix from volatilities and correlation coefficients
 * 
 * @param {Array<number>} volatilities - Array of asset volatilities
 * @param {Array<Array<number>>} correlationMatrix - Correlation matrix
 * @returns {Array<Array<number>>} Covariance matrix
 */
function correlationToCovariance(volatilities, correlationMatrix) {
    const n = volatilities.length;
    const covarianceMatrix = [];

    for (let i = 0; i < n; i++) {
        covarianceMatrix[i] = [];
        for (let j = 0; j < n; j++) {
            covarianceMatrix[i][j] = volatilities[i] * volatilities[j] * correlationMatrix[i][j];
        }
    }

    return covarianceMatrix;
}

/**
 * Perform Cholesky decomposition for correlated random variables
 * 
 * @param {Array<Array<number>>} matrix - Positive definite matrix
 * @returns {Array<Array<number>>} Lower triangular matrix L where A = L * L^T
 */
function choleskyDecomposition(matrix) {
    const n = matrix.length;
    const L = Array(n).fill().map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            if (i === j) {
                // Diagonal elements
                let sum = 0;
                for (let k = 0; k < j; k++) {
                    sum += L[i][k] * L[i][k];
                }
                L[i][j] = Math.sqrt(matrix[i][i] - sum);
            } else {
                // Non-diagonal elements
                let sum = 0;
                for (let k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (matrix[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

/**
 * Generate correlated random variables using Cholesky decomposition
 * 
 * @param {number} size - Number of samples to generate
 * @param {Array<Array<number>>} covarianceMatrix - Covariance matrix
 * @returns {Array<Array<number>>} Array of correlated random vectors
 */
function generateCorrelatedNormals(size, covarianceMatrix) {
    const n = covarianceMatrix.length;
    const L = choleskyDecomposition(covarianceMatrix);
    const result = [];

    for (let i = 0; i < size; i++) {
        // Generate independent standard normals
        const independent = randomNormalArray(n, 0, 1);

        // Transform to correlated variables: Y = L * X
        const correlated = Array(n).fill(0);
        for (let j = 0; j < n; j++) {
            for (let k = 0; k <= j; k++) {
                correlated[j] += L[j][k] * independent[k];
            }
        }

        result.push(correlated);
    }

    return result;
}

/**
 * Calculate basic statistical measures
 * 
 * @param {Array<number>} data - Array of numerical data
 * @returns {Object} Statistical measures (mean, std, min, max, etc.)
 */
function calculateStatistics(data) {
    if (data.length === 0) {
        return { mean: 0, std: 0, min: 0, max: 0, median: 0 };
    }

    const sorted = [...data].sort((a, b) => a - b);
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;

    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    const std = Math.sqrt(variance);

    const median = sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)];

    return {
        mean,
        std,
        variance,
        min: sorted[0],
        max: sorted[sorted.length - 1],
        median,
        count: data.length
    };
}

/**
 * Bootstrap sampling - sample with replacement
 * 
 * @param {Array<number>} data - Original data array
 * @param {number} sampleSize - Size of bootstrap sample
 * @returns {Array<number>} Bootstrap sample
 */
function bootstrapSample(data, sampleSize = data.length) {
    const sample = [];
    for (let i = 0; i < sampleSize; i++) {
        const randomIndex = Math.floor(Math.random() * data.length);
        sample.push(data[randomIndex]);
    }
    return sample;
}

module.exports = {
    randomNormal,
    randomNormalArray,
    calculatePercentile,
    calculateVaR,
    calculateCVaR,
    calculatePortfolioStats,
    correlationToCovariance,
    choleskyDecomposition,
    generateCorrelatedNormals,
    calculateStatistics,
    bootstrapSample
};
