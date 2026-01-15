#ifndef MLRISK_SIZING_H
#define MLRISK_SIZING_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file sizing.h
 * @brief Position sizing based on volatility targeting and risk capping
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
 * @param target_vol Target per-period volatility (scalar)
 * @param equity Account equity (scalar)
 * @param price Asset price (array of length n)
 * @param max_leverage Maximum leverage multiplier (e.g., 2.0 for 2x)
 * @param n Length of arrays
 * @param position_out Output array of position sizes (length n, must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status vol_target_position(
    const double *sigma,
    double target_vol,
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
);

/**
 * @brief Compute position size using risk capping
 * 
 * Computes position size such that the maximum risk per position is capped.
 * Formula: position = (risk_cap / sigma) * (equity / price)
 * 
 * If sigma is NaN or <= 0, returns 0.
 * The notional exposure (position * price) is capped by max_leverage * equity.
 * 
 * @param sigma Per-period volatility forecast (array of length n)
 * @param risk_cap Maximum risk per position as fraction of equity (scalar)
 * @param equity Account equity (scalar)
 * @param price Asset price (array of length n)
 * @param max_leverage Maximum leverage multiplier (e.g., 2.0 for 2x)
 * @param n Length of arrays
 * @param position_out Output array of position sizes (length n, must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status risk_cap_position(
    const double *sigma,
    double risk_cap,
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_SIZING_H */
