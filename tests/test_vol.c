#include "mlrisk/vol.h"
#include <stdio.h>
#include <math.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

#define TOLERANCE 1e-9

static int test_parkinson_known_answer(void) {
    // high = 100*e^0.02, low = 100 -> sigma = 0.02 / sqrt(4 ln 2)
    double high[] = {100.0 * exp(0.02), 100.0, 99.0};
    double low[] = {100.0, 100.0, 100.0}; // bar 2: high < low -> NAN
    double out[3];

    mlr_status status = mlr_parkinson_vol(high, low, 3, out);
    ASSERT(status == MLR_OK, "parkinson should return MLR_OK");

    double expected = 0.02 / sqrt(4.0 * log(2.0));
    ASSERT(fabs(out[0] - expected) < TOLERANCE, "parkinson known answer");
    ASSERT(out[1] == 0.0, "zero range should give zero vol");
    ASSERT(mlr_isnan(out[2]), "high < low should give NAN");

    ASSERT(mlr_parkinson_vol(NULL, low, 3, out) == MLR_EINVAL, "NULL high -> MLR_EINVAL");
    ASSERT(mlr_parkinson_vol(high, low, 0, out) == MLR_EINVAL, "n=0 -> MLR_EINVAL");

    printf("  PASS: parkinson known answer\n");
    return 0;
}

static int test_garman_klass_known_answer(void) {
    // Bar 0: open == close kills the second term -> sigma = sqrt(0.5)*ln(h/l)
    // Bar 1: tiny range, big close-open move -> negative sigma2 -> NAN
    // Bar 2: zero price -> NAN
    double open[] = {100.0, 100.0, 100.0};
    double high[] = {102.0, 100.5, 102.0};
    double low[] = {100.0, 99.9, 100.0};
    double close[] = {100.0, 110.0, 0.0};
    double out[3];

    mlr_status status = mlr_garman_klass_vol(open, high, low, close, 3, out);
    ASSERT(status == MLR_OK, "garman-klass should return MLR_OK");

    double expected = sqrt(0.5) * log(102.0 / 100.0);
    ASSERT(fabs(out[0] - expected) < TOLERANCE, "garman-klass known answer");
    ASSERT(mlr_isnan(out[1]), "negative sigma2 should give NAN");
    ASSERT(mlr_isnan(out[2]), "non-positive close should give NAN");

    ASSERT(mlr_garman_klass_vol(NULL, high, low, close, 3, out) == MLR_EINVAL,
           "NULL open -> MLR_EINVAL");

    printf("  PASS: garman-klass known answer\n");
    return 0;
}

static int test_garch_filter_known_answer(void) {
    // Hand-computed 3-step recursion, seed = mean of squared returns
    mlr_garch model = {0.2, 0.3, 0.5, 0.0, 0.0, 1};
    double returns[] = {1.0, -2.0, 3.0};
    double sigma[3];

    mlr_status status = mlr_garch_filter(&model, returns, 3, sigma);
    ASSERT(status == MLR_OK, "garch_filter should return MLR_OK");

    double var0 = (1.0 + 4.0 + 9.0) / 3.0;                    // 14/3
    double s2_1 = 0.2 + 0.3 * 1.0 + 0.5 * var0;               // 0.5 + 7/3
    double s2_2 = 0.2 + 0.3 * 4.0 + 0.5 * s2_1;
    ASSERT(fabs(sigma[0] - sqrt(var0)) < 1e-12, "filter step 0");
    ASSERT(fabs(sigma[1] - sqrt(s2_1)) < 1e-12, "filter step 1");
    ASSERT(fabs(sigma[2] - sqrt(s2_2)) < 1e-12, "filter step 2");

    printf("  PASS: garch filter known answer\n");
    return 0;
}

static int test_garch_forecast(void) {
    mlr_garch model = {0.2, 0.3, 0.5, 0.0, 0.0, 1};
    model.sigma2_next = 2.5;
    double sigma[1000];

    mlr_status status = mlr_garch_forecast(&model, 1000, sigma);
    ASSERT(status == MLR_OK, "garch_forecast should return MLR_OK");

    // h=1 is exactly sqrt(sigma2_next)
    ASSERT(fabs(sigma[0] - sqrt(2.5)) < 1e-12, "forecast h=1 equals sigma2_next");

    // Long horizon converges to unconditional vol: sqrt(omega/(1-alpha-beta))
    double uncond = sqrt(0.2 / (1.0 - 0.8));
    ASSERT(fabs(sigma[999] - uncond) < 1e-6, "forecast converges to unconditional vol");

    ASSERT(mlr_garch_forecast(&model, 0, sigma) == MLR_EINVAL, "horizon=0 -> MLR_EINVAL");
    model.alpha = -0.1;
    ASSERT(mlr_garch_forecast(&model, 1, sigma) == MLR_EINVAL, "bad params -> MLR_EINVAL");

    printf("  PASS: garch forecast\n");
    return 0;
}

// Deterministic LCG + Box-Muller so the simulation is identical on every platform
static double garch_lcg_u01(unsigned long long *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(*state >> 11) + 0.5) / 9007199254740992.0;
}

static double garch_gauss(unsigned long long *state) {
    double u1 = garch_lcg_u01(state);
    double u2 = garch_lcg_u01(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// Same NLL the fitter minimizes (constants dropped), replicated independently
static double test_nll(const double *r, size_t n, double omega, double alpha, double beta) {
    double var0 = 0.0;
    for (size_t t = 0; t < n; t++) var0 += r[t] * r[t];
    var0 /= (double)n;

    double s2 = var0;
    double nll = 0.0;
    for (size_t t = 0; t < n; t++) {
        nll += log(s2) + (r[t] * r[t]) / s2;
        s2 = omega + alpha * r[t] * r[t] + beta * s2;
    }
    return 0.5 * nll;
}

static int test_garch_fit_recovers_simulation(void) {
    enum { N = 2000 };
    static double returns[N];
    const double true_omega = 2e-6, true_alpha = 0.10, true_beta = 0.85;

    unsigned long long state = 12345;
    double s2 = true_omega / (1.0 - true_alpha - true_beta);
    for (size_t t = 0; t < N; t++) {
        returns[t] = sqrt(s2) * garch_gauss(&state);
        s2 = true_omega + true_alpha * returns[t] * returns[t] + true_beta * s2;
    }

    mlr_garch model;
    mlr_status status = mlr_garch_fit(returns, N, &model);
    ASSERT(status == MLR_OK, "garch_fit should return MLR_OK");

    // Constraints hold
    ASSERT(model.omega > 0.0, "fitted omega should be positive");
    ASSERT(model.alpha >= 0.0 && model.beta >= 0.0, "fitted alpha/beta non-negative");
    double persistence = model.alpha + model.beta;
    ASSERT(persistence > 0.8 && persistence < 0.9999, "fitted persistence in (0.8, 0.9999)");

    // The robust cross-compiler invariant: the fit is at least as good as the truth
    double nll_fit = test_nll(returns, N, model.omega, model.alpha, model.beta);
    double nll_true = test_nll(returns, N, true_omega, true_alpha, true_beta);
    ASSERT(nll_fit <= nll_true + 1e-9, "fitted NLL should not exceed true-parameter NLL");
    ASSERT(fabs(model.loglik - (-nll_fit)) < 1e-6 * fabs(nll_fit),
           "loglik field should match the fitted NLL");

    // Loose recovery bounds
    ASSERT(fabs(model.alpha - true_alpha) < 0.08, "alpha recovered within loose bounds");

    // sigma2_next is a plausible variance
    ASSERT(model.sigma2_next > 0.0 && model.sigma2_next < 100.0 * s2,
           "sigma2_next should be a plausible variance");

    printf("  PASS: garch fit recovers simulated parameters\n");
    return 0;
}

static int test_garch_fit_degenerate_inputs(void) {
    double zeros[25] = {0.0};
    double small[5] = {0.01, -0.01, 0.02, 0.0, 0.01};
    mlr_garch model;

    ASSERT(mlr_garch_fit(zeros, 25, &model) == MLR_EDOMAIN,
           "constant-zero returns should return MLR_EDOMAIN");
    ASSERT(mlr_garch_fit(small, 5, &model) == MLR_EINVAL, "n < 20 should return MLR_EINVAL");
    ASSERT(mlr_garch_fit(NULL, 25, &model) == MLR_EINVAL, "NULL returns -> MLR_EINVAL");
    ASSERT(mlr_garch_fit(zeros, 25, NULL) == MLR_EINVAL, "NULL model -> MLR_EINVAL");

    printf("  PASS: garch fit degenerate inputs\n");
    return 0;
}

int test_vol(void) {
    int failures = 0;
    failures += test_parkinson_known_answer();
    failures += test_garman_klass_known_answer();
    failures += test_garch_filter_known_answer();
    failures += test_garch_forecast();
    failures += test_garch_fit_recovers_simulation();
    failures += test_garch_fit_degenerate_inputs();
    return failures;
}
