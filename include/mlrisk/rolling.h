#ifndef MLRISK_ROLLING_H
#define MLRISK_ROLLING_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file rolling.h
 * @brief Rolling statistics and EWMA volatility computation
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute rolling mean of a time series
 * 
 * For each index i, computes the mean of x[i-window+1] to x[i] (inclusive).
 * If i < window-1, sets out[i] = MLR_NAN.
 *
 * Windows containing a non-finite value (NaN/Inf, e.g. missing data) emit
 * MLR_NAN; output recovers as soon as the bad value leaves the window.
 * 
 * @param x Input array of length n
 * @param n Length of input array
 * @param window Window size (must be >= 1)
 * @param out Output array of length n (must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_rolling_mean(const double *x, size_t n, size_t window, double *out);

/**
 * @brief Compute rolling standard deviation of a time series
 * 
 * O(n) sliding Welford updates (numerically stable). For each index i, computes
 * the std dev of x[i-window+1] to x[i] (inclusive). If i < window-1, sets
 * out[i] = MLR_NAN.
 *
 * Uses population variance (divides by window, not window-1); note this differs
 * from pandas rolling().std(), which defaults to the sample convention.
 *
 * Windows containing a non-finite value (NaN/Inf, e.g. missing data) emit
 * MLR_NAN; output recovers as soon as the bad value leaves the window.
 * 
 * @param x Input array of length n
 * @param n Length of input array
 * @param window Window size (must be >= 1)
 * @param out Output array of length n (must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_rolling_std(const double *x, size_t n, size_t window, double *out);

/**
 * @brief Compute EWMA (Exponentially Weighted Moving Average) volatility
 * 
 * Computes per-period volatility using EWMA:
 *   sigma[t]^2 = lambda * sigma[t-1]^2 + (1-lambda) * returns[t]^2
 * 
 * For the first period, uses returns[0]^2 as initial variance.
 *
 * Note: the recursion carries state, so a non-finite return poisons all
 * subsequent outputs. Clean your return series before calling.
 *
 * @param returns Array of returns (length n)
 * @param n Length of returns array
 * @param lambda Decay factor (typically 0.94-0.97, must be in [0, 1])
 * @param out Output array of per-period sigma values (length n, must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_ewma_vol(const double *returns, size_t n, double lambda, double *out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_ROLLING_H */
