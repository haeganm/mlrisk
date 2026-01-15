#include "mlrisk/sizing.h"
#include <math.h>

mlr_status vol_target_position(
    const double *sigma,
    double target_vol,
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
) {
    if (sigma == NULL || price == NULL || position_out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0) {
        return MLR_EINVAL;
    }
    if (equity <= 0.0 || max_leverage <= 0.0) {
        return MLR_EINVAL;
    }

    double max_notional = max_leverage * equity;

    for (size_t i = 0; i < n; i++) {
        if (mlr_isnan(sigma[i]) || sigma[i] <= 0.0 || price[i] <= 0.0) {
            position_out[i] = 0.0;
            continue;
        }

        // Compute target position: (target_vol / sigma) * (equity / price)
        double position = (target_vol / sigma[i]) * (equity / price[i]);

        // Cap by maximum notional exposure
        double notional = position * price[i];
        if (notional > max_notional) {
            position = max_notional / price[i];
        }

        position_out[i] = position;
    }

    return MLR_OK;
}

mlr_status risk_cap_position(
    const double *sigma,
    double risk_cap,
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
) {
    if (sigma == NULL || price == NULL || position_out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0) {
        return MLR_EINVAL;
    }
    if (equity <= 0.0 || max_leverage <= 0.0 || risk_cap <= 0.0) {
        return MLR_EINVAL;
    }

    double max_notional = max_leverage * equity;

    for (size_t i = 0; i < n; i++) {
        if (mlr_isnan(sigma[i]) || sigma[i] <= 0.0 || price[i] <= 0.0) {
            position_out[i] = 0.0;
            continue;
        }

        // Compute position: (risk_cap / sigma) * (equity / price)
        double position = (risk_cap / sigma[i]) * (equity / price[i]);

        // Cap by maximum notional exposure
        double notional = position * price[i];
        if (notional > max_notional) {
            position = max_notional / price[i];
        }

        position_out[i] = position;
    }

    return MLR_OK;
}
