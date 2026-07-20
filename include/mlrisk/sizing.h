#ifndef MLRISK_SIZING_H
#define MLRISK_SIZING_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file sizing.h
 * @brief Position sizing based on volatility targeting
 *
 * All volatility values (sigma, target_vol) are per-period.
 * To convert annualized volatility to per-period: per_period = annualized / sqrt(periods_per_year)
 * For daily data: per_period = annualized / sqrt(252)
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute position size using volatility targeting
 * 
 * Computes position size such that the portfolio volatility matches target_vol.
 * Formula: position = (target_vol / sigma) * (equity / price)
 * 
 * If sigma is NaN or <= 0, returns 0.
 * The notional exposure (position * price) is capped by max_leverage * equity.
 * 
 * @param sigma Per-period volatility forecast (array of length n)
 * @param target_vol Target per-period volatility (scalar, must be finite and > 0).
 *                   To cap risk per position instead, pass the risk cap (as a
 *                   fraction of equity) here - the formula is identical.
 * @param equity Account equity (scalar)
 * @param price Asset price (array of length n)
 * @param max_leverage Maximum leverage multiplier (e.g., 2.0 for 2x)
 * @param n Length of arrays
 * @param position_out Output array of position sizes (length n, must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_vol_target_position(
    const double *sigma,
    double target_vol,
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
);

/**
 * @brief Kelly fraction from a returns sample
 *
 * f = fraction * mean(returns) / sample_variance(returns)
 * (sample variance uses the n-1 denominator; this is the continuous
 * mean-variance Kelly approximation).
 *
 * f_out may be negative when the sample edge is negative - the caller decides
 * how to act on that (typically: no position).
 *
 * @param returns Per-period returns (length n, all finite)
 * @param n Number of returns (must be >= 2)
 * @param fraction Kelly multiplier: 1.0 = full Kelly, 0.5 = half Kelly (must be > 0)
 * @param f_out Receives the (possibly negative) Kelly fraction of equity
 * @return MLR_OK on success, MLR_EINVAL on invalid input,
 *         MLR_EDOMAIN if returns have zero variance
 */
mlr_status mlr_kelly_fraction(const double *returns, size_t n, double fraction, double *f_out);

/**
 * @brief Drawdown-based exposure scaling
 *
 * For each i, with running peak P_i = max(equity[0..i]) and drawdown
 * dd_i = 1 - equity[i]/P_i:
 *
 *   scale_out[i] = clamp(1 - dd_i / max_dd, 0, 1)
 *
 * Full exposure at zero drawdown, tapering linearly to zero exposure at
 * dd >= max_dd. Multiply your position sizes by scale_out.
 *
 * @param equity Cumulative equity path (length n, all finite and > 0)
 * @param n Number of samples
 * @param max_dd Drawdown at which exposure reaches zero (in (0, 1])
 * @param scale_out Output scale factors in [0, 1] (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input,
 *         MLR_EDOMAIN if any equity value is non-finite or <= 0
 */
mlr_status mlr_drawdown_scale(const double *equity, size_t n, double max_dd, double *scale_out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_SIZING_H */
