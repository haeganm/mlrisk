#ifndef MLRISK_VOL_H
#define MLRISK_VOL_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file vol.h
 * @brief Volatility estimators: range-based (Parkinson, Garman-Klass)
 *
 * All outputs are per-period volatility (sigma), matching the rest of the
 * library. See the EWMA estimator in rolling.h.
 */

#ifdef __cplusplus
extern "C" {
#endif

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
