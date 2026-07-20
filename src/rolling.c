#include "mlrisk/rolling.h"
#include <math.h>

// Reference level for offset-shifted accumulation: the first finite value.
// Shifting keeps precision when the series has a large common level
// (prices, index values); variance is shift-invariant and the mean shifts back.
static double first_finite(const double *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (mlr_isfinite(x[i])) {
            return x[i];
        }
    }
    return 0.0;
}

mlr_status mlr_rolling_mean(const double *x, size_t n, size_t window, double *out) {
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

    double offset = first_finite(x, n);

    // O(n) sliding sum over the finite values only; windows containing any
    // non-finite value emit NAN and the accumulator stays clean, so output
    // recovers as soon as the bad value leaves the window
    double sum = 0.0;
    size_t bad = 0;
    for (size_t j = 0; j < window; j++) {
        if (mlr_isfinite(x[j])) {
            sum += x[j] - offset;
        } else {
            bad++;
        }
    }

    for (size_t i = window - 1; ; i++) {
        out[i] = (bad > 0) ? MLR_NAN : offset + sum / (double)window;
        if (i + 1 >= n) {
            break;
        }
        if (mlr_isfinite(x[i + 1])) {
            sum += x[i + 1] - offset;
        } else {
            bad++;
        }
        if (mlr_isfinite(x[i + 1 - window])) {
            sum -= x[i + 1 - window] - offset;
        } else {
            bad--;
        }
    }

    return MLR_OK;
}

mlr_status mlr_rolling_std(const double *x, size_t n, size_t window, double *out) {
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
            out[i] = mlr_isfinite(x[i]) ? 0.0 : MLR_NAN;
        }
        return MLR_OK;
    }

    double w = (double)window;
    double offset = first_finite(x, n);

    size_t bad = 0;
    for (size_t j = 0; j < window; j++) {
        if (!mlr_isfinite(x[j])) {
            bad++;
        }
    }

    // O(n) rolling Welford updates while the window is clean; windows with a
    // non-finite value emit NAN, and the accumulators are rebuilt (O(window),
    // only on recovery) once the bad value leaves
    double mean = 0.0;
    double m2 = 0.0;
    int valid = 0;

    for (size_t i = window - 1; i < n; i++) {
        if (i >= window) {
            // Slide: x[i] entered, x[i - window] left
            if (!mlr_isfinite(x[i])) {
                bad++;
            }
            if (!mlr_isfinite(x[i - window])) {
                bad--;
            }
        }

        if (bad > 0) {
            out[i] = MLR_NAN;
            valid = 0;
            continue;
        }

        if (!valid) {
            // Rebuild with standard Welford accumulation over this window
            mean = 0.0;
            m2 = 0.0;
            for (size_t j = i - window + 1; j <= i; j++) {
                double xs = x[j] - offset;
                double delta = xs - mean;
                mean += delta / (double)(j - (i - window + 1) + 1);
                m2 += delta * (xs - mean);
            }
            valid = 1;
        } else {
            // Remove the oldest sample, add the newest (rolling Welford)
            double x_old = x[i - window] - offset;
            double x_new = x[i] - offset;

            double mean_removed = (w * mean - x_old) / (w - 1.0);
            m2 -= (x_old - mean) * (x_old - mean_removed);
            mean = mean_removed;

            double mean_added = mean + (x_new - mean) / w;
            m2 += (x_new - mean) * (x_new - mean_added);
            mean = mean_added;
        }

        // Removal can push m2 epsilon-negative
        out[i] = sqrt(fmax(m2, 0.0) / w);
    }

    return MLR_OK;
}

mlr_status mlr_ewma_vol(const double *returns, size_t n, double lambda, double *out) {
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
