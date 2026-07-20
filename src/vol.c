#include "mlrisk/vol.h"
#include <math.h>

#define LN2 0.69314718055994530942

/* ---------------- GARCH(1,1) ---------------- */

static int garch_params_valid(double omega, double alpha, double beta) {
    return omega > 1e-16 && alpha >= 0.0 && beta >= 0.0 && alpha + beta < 0.9999;
}

// Gaussian negative log-likelihood (constants dropped), recursion seeded with var0.
// Infeasible parameters return HUGE_VAL (penalty constraints for the optimizer).
static double garch_nll(const double *r, size_t n, double var0,
                        double omega, double alpha, double beta) {
    if (!garch_params_valid(omega, alpha, beta)) {
        return HUGE_VAL;
    }

    double s2 = var0;
    double nll = 0.0;
    for (size_t t = 0; t < n; t++) {
        nll += log(s2) + (r[t] * r[t]) / s2;
        s2 = omega + alpha * r[t] * r[t] + beta * s2;
    }
    return 0.5 * nll;
}

typedef struct {
    const double *r;
    size_t n;
    double var0;
} garch_ctx;

static double nm_eval(const garch_ctx *ctx, const double x[3]) {
    return garch_nll(ctx->r, ctx->n, ctx->var0, x[0], x[1], x[2]);
}

// Nelder-Mead in 3 dimensions with standard coefficients
// (reflection 1, expansion 2, contraction 0.5, shrink 0.5).
// Returns 1 if converged, 0 if it hit the iteration cap.
static int nelder_mead3(const garch_ctx *ctx, double x[3], double *f_out) {
    enum { DIM = 3, PTS = 4 };
    double simplex[PTS][DIM];
    double f[PTS];

    for (int i = 0; i < PTS; i++) {
        for (int j = 0; j < DIM; j++) {
            simplex[i][j] = x[j];
        }
        if (i > 0) {
            int j = i - 1;
            simplex[i][j] += (simplex[i][j] != 0.0) ? 0.1 * simplex[i][j] : 1e-6;
        }
        f[i] = nm_eval(ctx, simplex[i]);
    }

    int converged = 0;
    for (int iter = 0; iter < 500; iter++) {
        // Sort simplex points by f (4 points: insertion sort)
        for (int i = 1; i < PTS; i++) {
            double fi = f[i];
            double xi[DIM];
            for (int j = 0; j < DIM; j++) xi[j] = simplex[i][j];
            int k = i - 1;
            while (k >= 0 && f[k] > fi) {
                f[k + 1] = f[k];
                for (int j = 0; j < DIM; j++) simplex[k + 1][j] = simplex[k][j];
                k--;
            }
            f[k + 1] = fi;
            for (int j = 0; j < DIM; j++) simplex[k + 1][j] = xi[j];
        }

        if (f[PTS - 1] - f[0] <= 1e-10 * (fabs(f[0]) + 1e-10)) {
            converged = 1;
            break;
        }

        // Centroid of the best three points
        double c[DIM];
        for (int j = 0; j < DIM; j++) {
            c[j] = (simplex[0][j] + simplex[1][j] + simplex[2][j]) / 3.0;
        }

        double xr[DIM], xe[DIM], xc[DIM];
        for (int j = 0; j < DIM; j++) xr[j] = c[j] + (c[j] - simplex[PTS - 1][j]);
        double fr = nm_eval(ctx, xr);

        if (fr < f[0]) {
            // Try to expand
            for (int j = 0; j < DIM; j++) xe[j] = c[j] + 2.0 * (c[j] - simplex[PTS - 1][j]);
            double fe = nm_eval(ctx, xe);
            if (fe < fr) {
                for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xe[j];
                f[PTS - 1] = fe;
            } else {
                for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xr[j];
                f[PTS - 1] = fr;
            }
        } else if (fr < f[PTS - 2]) {
            for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xr[j];
            f[PTS - 1] = fr;
        } else {
            // Contract (outside if reflection improved on the worst, else inside)
            double sign = (fr < f[PTS - 1]) ? 0.5 : -0.5;
            for (int j = 0; j < DIM; j++) xc[j] = c[j] + sign * (c[j] - simplex[PTS - 1][j]);
            double fc = nm_eval(ctx, xc);
            double fbar = (fr < f[PTS - 1]) ? fr : f[PTS - 1];
            if (fc <= fbar) {
                for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xc[j];
                f[PTS - 1] = fc;
            } else {
                // Shrink toward the best point
                for (int i = 1; i < PTS; i++) {
                    for (int j = 0; j < DIM; j++) {
                        simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                    }
                    f[i] = nm_eval(ctx, simplex[i]);
                }
            }
        }
    }

    // Best point (sorted on break; re-scan to be safe after shrink paths)
    int best = 0;
    for (int i = 1; i < PTS; i++) {
        if (f[i] < f[best]) best = i;
    }
    for (int j = 0; j < DIM; j++) x[j] = simplex[best][j];
    *f_out = f[best];
    return converged;
}

mlr_status mlr_garch_fit(const double *returns, size_t n, mlr_garch *model_out) {
    if (returns == NULL || model_out == NULL) {
        return MLR_EINVAL;
    }
    if (n < 20) {
        return MLR_EINVAL;
    }

    double var0 = 0.0;
    for (size_t t = 0; t < n; t++) {
        if (!mlr_isfinite(returns[t])) {
            return MLR_EINVAL;
        }
        var0 += returns[t] * returns[t];
    }
    var0 /= (double)n;
    if (var0 <= 0.0) {
        return MLR_EDOMAIN;
    }

    garch_ctx ctx = {returns, n, var0};

    // Coarse feasible grid, omega from variance targeting
    static const double alphas[] = {0.02, 0.05, 0.10, 0.15};
    static const double betas[] = {0.80, 0.88, 0.94};
    double best[3] = {var0 * 0.1, 0.05, 0.85};
    double best_f = HUGE_VAL;

    for (size_t a = 0; a < 4; a++) {
        for (size_t b = 0; b < 3; b++) {
            if (alphas[a] + betas[b] >= 0.9999) {
                continue;
            }
            double x[3] = {var0 * (1.0 - alphas[a] - betas[b]), alphas[a], betas[b]};
            double fx = nm_eval(&ctx, x);
            if (fx < best_f) {
                best_f = fx;
                best[0] = x[0];
                best[1] = x[1];
                best[2] = x[2];
            }
        }
    }

    double f_min = best_f;
    int converged = nelder_mead3(&ctx, best, &f_min);

    model_out->omega = best[0];
    model_out->alpha = best[1];
    model_out->beta = best[2];
    model_out->loglik = -f_min;
    model_out->converged = converged;

    // Run the recursion once more to get the one-step-ahead variance
    double s2 = var0;
    for (size_t t = 0; t < n; t++) {
        s2 = best[0] + best[1] * returns[t] * returns[t] + best[2] * s2;
    }
    model_out->sigma2_next = s2;

    return MLR_OK;
}

mlr_status mlr_garch_filter(const mlr_garch *model, const double *returns, size_t n,
                            double *sigma_out) {
    if (model == NULL || returns == NULL || sigma_out == NULL || n == 0) {
        return MLR_EINVAL;
    }
    if (!garch_params_valid(model->omega, model->alpha, model->beta)) {
        return MLR_EINVAL;
    }

    double var0 = 0.0;
    for (size_t t = 0; t < n; t++) {
        var0 += returns[t] * returns[t];
    }
    var0 /= (double)n;

    double s2 = var0;
    for (size_t t = 0; t < n; t++) {
        sigma_out[t] = sqrt(s2);
        s2 = model->omega + model->alpha * returns[t] * returns[t] + model->beta * s2;
    }

    return MLR_OK;
}

mlr_status mlr_garch_forecast(const mlr_garch *model, size_t horizon, double *sigma_out) {
    if (model == NULL || sigma_out == NULL || horizon == 0) {
        return MLR_EINVAL;
    }
    if (!garch_params_valid(model->omega, model->alpha, model->beta)) {
        return MLR_EINVAL;
    }
    if (model->sigma2_next < 0.0 || !mlr_isfinite(model->sigma2_next)) {
        return MLR_EINVAL;
    }

    double persistence = model->alpha + model->beta;
    double uncond = model->omega / (1.0 - persistence);
    double decay = 1.0;

    for (size_t h = 0; h < horizon; h++) {
        double s2 = uncond + decay * (model->sigma2_next - uncond);
        sigma_out[h] = sqrt(s2);
        decay *= persistence;
    }

    return MLR_OK;
}

/* ---------------- Range-based estimators ---------------- */

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
