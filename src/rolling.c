#include "mlrisk/rolling.h"
#include <math.h>

mlr_status rolling_mean(const double *x, size_t n, size_t window, double *out) {
    if (x == NULL || out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0 || window == 0) {
        return MLR_EINVAL;
    }

    // For indices < window-1, set to NAN
    for (size_t i = 0; i < window - 1 && i < n; i++) {
        out[i] = MLR_NAN;
    }

    // Compute rolling mean for valid windows
    for (size_t i = window - 1; i < n; i++) {
        double sum = 0.0;
        for (size_t j = i - window + 1; j <= i; j++) {
            sum += x[j];
        }
        out[i] = sum / window;
    }

    return MLR_OK;
}

mlr_status rolling_std(const double *x, size_t n, size_t window, double *out) {
    if (x == NULL || out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0 || window == 0) {
        return MLR_EINVAL;
    }

    // For indices < window-1, set to NAN
    for (size_t i = 0; i < window - 1 && i < n; i++) {
        out[i] = MLR_NAN;
    }

    // Compute rolling std using Welford's algorithm adapted for fixed window
    for (size_t i = window - 1; i < n; i++) {
        // Compute mean first
        double mean = 0.0;
        for (size_t j = i - window + 1; j <= i; j++) {
            mean += x[j];
        }
        mean /= window;

        // Compute variance
        double variance = 0.0;
        for (size_t j = i - window + 1; j <= i; j++) {
            double diff = x[j] - mean;
            variance += diff * diff;
        }
        variance /= window;

        out[i] = sqrt(variance);
    }

    return MLR_OK;
}

mlr_status ewma_vol(const double *returns, size_t n, double lambda, double *out) {
    if (returns == NULL || out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0) {
        return MLR_EINVAL;
    }
    if (lambda < 0.0 || lambda > 1.0) {
        return MLR_EINVAL;
    }

    // Initialize first variance with first return squared
    double variance = returns[0] * returns[0];
    out[0] = sqrt(variance);

    // Update variance using EWMA recursion
    for (size_t i = 1; i < n; i++) {
        variance = lambda * variance + (1.0 - lambda) * (returns[i] * returns[i]);
        out[i] = sqrt(variance);
    }

    return MLR_OK;
}
