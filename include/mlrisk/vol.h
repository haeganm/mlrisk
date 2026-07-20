#ifndef MLRISK_VOL_H
#define MLRISK_VOL_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file vol.h
 * @brief Volatility estimators: GARCH(1,1) and range-based (Parkinson, Garman-Klass)
 *
 * All outputs are per-period volatility (sigma), matching the rest of the
 * library. See the EWMA estimator in rolling.h.
 *
 * Note on range estimators: with discretely sampled bars they carry a small
 * downward bias (the observed high/low understate the continuous extremes);
 * expect readings a few percent below true volatility on e.g. 390-tick bars.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GARCH(1,1) model: sigma2[t] = omega + alpha*r[t-1]^2 + beta*sigma2[t-1]
 *
 * Returns are assumed mean-zero (standard for financial returns at daily or
 * higher frequency). The variance recursion is seeded with the mean of
 * squared returns.
 */
typedef struct {
    double omega;        /**< Constant term (> 0) */
    double alpha;        /**< ARCH coefficient (>= 0) */
    double beta;         /**< GARCH coefficient (>= 0, alpha + beta < 1) */
    double sigma2_next;  /**< One-step-ahead conditional variance after the fit sample */
    double loglik;       /**< Maximized Gaussian log-likelihood (constants dropped) */
    int converged;       /**< 1 if the optimizer converged, 0 = best-effort result */
} mlr_garch;

/**
 * @brief Fit GARCH(1,1) by Gaussian maximum likelihood
 *
 * Dependency-free MLE: a coarse feasible grid seeds a Nelder-Mead refinement.
 * Non-convergence is not an error - the best point found is returned with
 * converged == 0.
 *
 * @param returns Mean-zero returns (length n, all finite)
 * @param n Number of returns (must be >= 20)
 * @param model_out Fitted model
 * @return MLR_OK on success, MLR_EINVAL on invalid input,
 *         MLR_EDOMAIN if returns have zero variance
 */
mlr_status mlr_garch_fit(const double *returns, size_t n, mlr_garch *model_out);

/**
 * @brief In-sample conditional volatility path
 *
 * sigma2[0] = mean of squared returns, then the GARCH recursion;
 * sigma_out[t] = sqrt(sigma2[t]).
 *
 * @param model Fitted (or manually constructed) model
 * @param returns Mean-zero returns (length n)
 * @param n Number of returns
 * @param sigma_out Output per-period sigma (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input or parameters
 */
mlr_status mlr_garch_filter(const mlr_garch *model, const double *returns, size_t n,
                            double *sigma_out);

/**
 * @brief Multi-step volatility forecast from the end of the fit sample
 *
 * sigma_out[h] is the per-period volatility h+1 steps ahead:
 * sigma2[T+1+h] = uncond + (alpha+beta)^h * (sigma2_next - uncond),
 * where uncond = omega / (1 - alpha - beta).
 *
 * @param model Fitted model
 * @param horizon Number of steps to forecast (>= 1)
 * @param sigma_out Output per-period sigma (length horizon, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input or parameters
 */
mlr_status mlr_garch_forecast(const mlr_garch *model, size_t horizon, double *sigma_out);

/**
 * @brief Parkinson range-based volatility, per bar
 *
 * sigma[i] = sqrt( ln(high[i]/low[i])^2 / (4 ln 2) )
 *
 * Bad bars (non-finite, <= 0, or high < low) produce out[i] = MLR_NAN;
 * the call still returns MLR_OK.
 *
 * @param high High prices (length n)
 * @param low Low prices (length n)
 * @param n Number of bars
 * @param out Output per-bar sigma (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_parkinson_vol(const double *high, const double *low, size_t n, double *out);

/**
 * @brief Garman-Klass range-based volatility, per bar
 *
 * sigma2[i] = 0.5 * ln(high/low)^2 - (2 ln 2 - 1) * ln(close/open)^2
 *
 * Bad bars (non-finite, <= 0, or high < low) produce out[i] = MLR_NAN, as do
 * bars where the estimator goes negative (a known Garman-Klass artifact);
 * the call still returns MLR_OK.
 *
 * @param open Open prices (length n)
 * @param high High prices (length n)
 * @param low Low prices (length n)
 * @param close Close prices (length n)
 * @param n Number of bars
 * @param out Output per-bar sigma (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_garman_klass_vol(const double *open, const double *high,
                                const double *low, const double *close,
                                size_t n, double *out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_VOL_H */
