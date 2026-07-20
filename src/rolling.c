#include "mlrisk/rolling.h"
#include <math.h>

mlr_status rolling_mean(const double *x, size_t n, size_t window, double *out) {
    if (x == NULL || out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0 || window == 0) {
        return MLR_EINVAL;
    }

    // Warmup: indices without a full window are NAN
    for (size_t i = 0; i < window - 1 && i < n; i++) {
        out[i] = MLR_NAN;
    }
    if (window > n) {
        return MLR_OK;
    }

    // O(n) sliding sum. Running-sum drift is negligible for return-scale data;
    // rolling_std uses the drift-resistant Welford path.
    double sum = 0.0;
    for (size_t j = 0; j < window; j++) {
        sum += x[j];
    }
    out[window - 1] = sum / (double)window;

    for (size_t i = window; i < n; i++) {
        sum += x[i] - x[i - window];
        out[i] = sum / (double)window;
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

    // Warmup: indices without a full window are NAN
    for (size_t i = 0; i < window - 1 && i < n; i++) {
        out[i] = MLR_NAN;
    }
    if (window > n) {
        return MLR_OK;
    }

    if (window == 1) {
        for (size_t i = 0; i < n; i++) {
            out[i] = 0.0;
        }
        return MLR_OK;
    }

    double w = (double)window;

    // Initialize first window with standard Welford accumulation
    double mean = 0.0;
    double m2 = 0.0;
    for (size_t j = 0; j < window; j++) {
        double delta = x[j] - mean;
        mean += delta / (double)(j + 1);
        m2 += delta * (x[j] - mean);
    }
    out[window - 1] = sqrt(fmax(m2, 0.0) / w);

    // Slide: remove the oldest sample, add the newest (rolling Welford updates)
    for (size_t i = window; i < n; i++) {
        double x_old = x[i - window];
        double x_new = x[i];

        double mean_removed = (w * mean - x_old) / (w - 1.0);
        m2 -= (x_old - mean) * (x_old - mean_removed);
        mean = mean_removed;

        double mean_added = mean + (x_new - mean) / w;
        m2 += (x_new - mean) * (x_new - mean_added);
        mean = mean_added;

        // Removal can push m2 epsilon-negative
        out[i] = sqrt(fmax(m2, 0.0) / w);
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
