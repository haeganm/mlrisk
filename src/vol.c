#include "mlrisk/vol.h"
#include <math.h>

#define LN2 0.69314718055994530942

static int bad_bar2(double high, double low) {
    return !mlr_isfinite(high) || !mlr_isfinite(low) ||
           high <= 0.0 || low <= 0.0 || high < low;
}

mlr_status mlr_parkinson_vol(const double *high, const double *low, size_t n, double *out) {
    if (high == NULL || low == NULL || out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < n; i++) {
        if (bad_bar2(high[i], low[i])) {
            out[i] = MLR_NAN;
            continue;
        }
        double hl = log(high[i] / low[i]);
        out[i] = sqrt(hl * hl / (4.0 * LN2));
    }

    return MLR_OK;
}

mlr_status mlr_garman_klass_vol(const double *open, const double *high,
                                const double *low, const double *close,
                                size_t n, double *out) {
    if (open == NULL || high == NULL || low == NULL || close == NULL || out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < n; i++) {
        if (bad_bar2(high[i], low[i]) ||
            !mlr_isfinite(open[i]) || open[i] <= 0.0 ||
            !mlr_isfinite(close[i]) || close[i] <= 0.0) {
            out[i] = MLR_NAN;
            continue;
        }
        double hl = log(high[i] / low[i]);
        double co = log(close[i] / open[i]);
        double sigma2 = 0.5 * hl * hl - (2.0 * LN2 - 1.0) * co * co;
        out[i] = sigma2 >= 0.0 ? sqrt(sigma2) : MLR_NAN;
    }

    return MLR_OK;
}
