#include "mlrisk/rolling.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

#define TOLERANCE 1e-9

static int test_rolling_mean_basic(void) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5];
    
    mlr_status status = mlr_rolling_mean(x, 5, 3, out);
    ASSERT(status == MLR_OK, "mlr_rolling_mean should return MLR_OK");
    
    // First two should be NAN
    ASSERT(mlr_isnan(out[0]), "out[0] should be NAN");
    ASSERT(mlr_isnan(out[1]), "out[1] should be NAN");
    
    // out[2] = mean of [1,2,3] = 2.0
    ASSERT(fabs(out[2] - 2.0) < TOLERANCE, "out[2] should be 2.0");
    
    // out[3] = mean of [2,3,4] = 3.0
    ASSERT(fabs(out[3] - 3.0) < TOLERANCE, "out[3] should be 3.0");
    
    // out[4] = mean of [3,4,5] = 4.0
    ASSERT(fabs(out[4] - 4.0) < TOLERANCE, "out[4] should be 4.0");
    
    printf("  PASS: mlr_rolling_mean basic\n");
    return 0;
}

static int test_rolling_mean_edge_cases(void) {
    double x[] = {1.0};
    double out[1];
    
    // Window = 1 should work
    mlr_status status = mlr_rolling_mean(x, 1, 1, out);
    ASSERT(status == MLR_OK, "mlr_rolling_mean with window=1 should work");
    ASSERT(fabs(out[0] - 1.0) < TOLERANCE, "out[0] should be 1.0");
    
    // Window > n should set all to NAN
    status = mlr_rolling_mean(x, 1, 5, out);
    ASSERT(status == MLR_OK, "mlr_rolling_mean with window>n should return OK");
    ASSERT(mlr_isnan(out[0]), "out[0] should be NAN when window>n");
    
    // NULL pointers
    status = mlr_rolling_mean(NULL, 1, 1, out);
    ASSERT(status == MLR_EINVAL, "mlr_rolling_mean with NULL x should return MLR_EINVAL");
    
    status = mlr_rolling_mean(x, 1, 1, NULL);
    ASSERT(status == MLR_EINVAL, "mlr_rolling_mean with NULL out should return MLR_EINVAL");
    
    // Zero window
    status = mlr_rolling_mean(x, 1, 0, out);
    ASSERT(status == MLR_EINVAL, "mlr_rolling_mean with window=0 should return MLR_EINVAL");
    
    printf("  PASS: mlr_rolling_mean edge cases\n");
    return 0;
}

static int test_rolling_std_basic(void) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5];
    
    mlr_status status = mlr_rolling_std(x, 5, 3, out);
    ASSERT(status == MLR_OK, "mlr_rolling_std should return MLR_OK");
    
    // First two should be NAN
    ASSERT(mlr_isnan(out[0]), "out[0] should be NAN");
    ASSERT(mlr_isnan(out[1]), "out[1] should be NAN");
    
    // out[2] = std of [1,2,3]
    // mean = 2.0, variance = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
    // std = sqrt(2/3) ≈ 0.8165
    double expected = sqrt(2.0 / 3.0);
    ASSERT(fabs(out[2] - expected) < TOLERANCE, "out[2] should match expected std");
    
    printf("  PASS: mlr_rolling_std basic\n");
    return 0;
}

static int test_ewma_vol_basic(void) {
    double returns[] = {0.01, -0.02, 0.015, -0.01, 0.02};
    double out[5];
    double lambda = 0.94;
    
    mlr_status status = mlr_ewma_vol(returns, 5, lambda, out);
    ASSERT(status == MLR_OK, "mlr_ewma_vol should return MLR_OK");
    
    // First value should be |returns[0]|
    ASSERT(fabs(out[0] - fabs(returns[0])) < TOLERANCE, "out[0] should be |returns[0]|");
    
    // Subsequent values should be positive
    for (size_t i = 1; i < 5; i++) {
        ASSERT(out[i] > 0.0, "EWMA vol should be positive");
    }
    
    printf("  PASS: mlr_ewma_vol basic\n");
    return 0;
}

static int test_ewma_vol_regime_change(void) {
    // Create synthetic data: low vol then high vol
    double returns[100];
    for (size_t i = 0; i < 50; i++) {
        returns[i] = 0.001 * ((double)rand() / RAND_MAX - 0.5); // Low vol
    }
    for (size_t i = 50; i < 100; i++) {
        returns[i] = 0.01 * ((double)rand() / RAND_MAX - 0.5); // High vol
    }
    
    double out[100];
    double lambda = 0.94;
    
    mlr_status status = mlr_ewma_vol(returns, 100, lambda, out);
    ASSERT(status == MLR_OK, "mlr_ewma_vol should return MLR_OK");
    
    // Volatility should increase after regime change
    double avg_early = 0.0, avg_late = 0.0;
    for (size_t i = 10; i < 50; i++) {
        avg_early += out[i];
    }
    for (size_t i = 60; i < 100; i++) {
        avg_late += out[i];
    }
    avg_early /= 40.0;
    avg_late /= 40.0;
    
    ASSERT(avg_late > avg_early, "EWMA should detect regime change");
    
    printf("  PASS: mlr_ewma_vol regime change\n");
    return 0;
}

static int test_ewma_vol_invalid_lambda(void) {
    double returns[] = {0.01};
    double out[1];
    
    mlr_status status = mlr_ewma_vol(returns, 1, -0.1, out);
    ASSERT(status == MLR_EINVAL, "mlr_ewma_vol with negative lambda should return MLR_EINVAL");
    
    status = mlr_ewma_vol(returns, 1, 1.5, out);
    ASSERT(status == MLR_EINVAL, "mlr_ewma_vol with lambda>1 should return MLR_EINVAL");
    
    printf("  PASS: mlr_ewma_vol invalid lambda\n");
    return 0;
}

// Deterministic LCG so results are identical on every platform
static double lcg_next(unsigned long long *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(*state >> 11) / 9007199254740992.0) - 0.5;
}

static int test_rolling_sliding_consistency(void) {
    enum { N = 1000 };
    static double x[N], fast_mean[N], fast_std[N];
    unsigned long long state = 42;
    for (size_t i = 0; i < N; i++) {
        x[i] = lcg_next(&state) * 0.05;
    }

    size_t windows[] = {2, 5, 50};
    for (size_t wi = 0; wi < 3; wi++) {
        size_t w = windows[wi];
        ASSERT(mlr_rolling_mean(x, N, w, fast_mean) == MLR_OK, "sliding mean OK");
        ASSERT(mlr_rolling_std(x, N, w, fast_std) == MLR_OK, "sliding std OK");

        for (size_t i = w - 1; i < N; i++) {
            double mean = 0.0;
            for (size_t j = i - w + 1; j <= i; j++) mean += x[j];
            mean /= (double)w;

            double var = 0.0;
            for (size_t j = i - w + 1; j <= i; j++) {
                double d = x[j] - mean;
                var += d * d;
            }
            var /= (double)w;

            ASSERT(fabs(fast_mean[i] - mean) < TOLERANCE, "sliding mean matches naive recompute");
            ASSERT(fabs(fast_std[i] - sqrt(var)) < TOLERANCE, "sliding std matches naive recompute");
        }
    }

    printf("  PASS: sliding algorithms match naive recompute\n");
    return 0;
}

static int test_rolling_std_edge_cases(void) {
    double constant[10] = {3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5};
    double out[10];

    // Constant input: variance must clamp to exactly 0
    ASSERT(mlr_rolling_std(constant, 10, 4, out) == MLR_OK, "constant std OK");
    for (size_t i = 3; i < 10; i++) {
        ASSERT(out[i] == 0.0, "std of constant array should be exactly 0");
    }

    // window > n: all NAN
    ASSERT(mlr_rolling_std(constant, 3, 5, out) == MLR_OK, "window>n std OK");
    for (size_t i = 0; i < 3; i++) {
        ASSERT(mlr_isnan(out[i]), "std should be NAN when window>n");
    }

    // window == 1: zeros
    ASSERT(mlr_rolling_std(constant, 3, 1, out) == MLR_OK, "window=1 std OK");
    ASSERT(out[0] == 0.0 && out[2] == 0.0, "std with window=1 should be 0");

    printf("  PASS: mlr_rolling_std edge cases\n");
    return 0;
}

static int test_rolling_shift_invariance(void) {
    // Std is shift-invariant and mean shifts exactly; a large common level
    // (prices, index values) must not destroy precision
    enum { N = 500, W = 20 };
    static double base[N], shifted[N], std_base[N], std_shifted[N];
    static double mean_base[N], mean_shifted[N];
    const double OFFSET = 1e9;

    unsigned long long state = 777;
    for (size_t i = 0; i < N; i++) {
        base[i] = lcg_next(&state);
        shifted[i] = base[i] + OFFSET;
    }

    ASSERT(mlr_rolling_std(base, N, W, std_base) == MLR_OK, "base std OK");
    ASSERT(mlr_rolling_std(shifted, N, W, std_shifted) == MLR_OK, "shifted std OK");
    ASSERT(mlr_rolling_mean(base, N, W, mean_base) == MLR_OK, "base mean OK");
    ASSERT(mlr_rolling_mean(shifted, N, W, mean_shifted) == MLR_OK, "shifted mean OK");

    for (size_t i = W - 1; i < N; i++) {
        ASSERT(fabs(std_shifted[i] - std_base[i]) < 1e-6,
               "std should be invariant under a 1e9 level shift");
        ASSERT(fabs((mean_shifted[i] - OFFSET) - mean_base[i]) < 1e-6,
               "mean should shift exactly under a 1e9 level shift");
    }

    printf("  PASS: shift invariance at offset 1e9\n");
    return 0;
}

static int test_rolling_nan_recovery(void) {
    // A NaN (missing data) must only affect windows containing it - the
    // sliding accumulators must recover once it leaves the window
    enum { N = 20, W = 3 };
    double x[N], mean_out[N], std_out[N];
    unsigned long long state = 4242;
    for (size_t i = 0; i < N; i++) {
        x[i] = lcg_next(&state);
    }
    x[7] = MLR_NAN;
    x[13] = INFINITY;

    ASSERT(mlr_rolling_mean(x, N, W, mean_out) == MLR_OK, "mean with NaN OK");
    ASSERT(mlr_rolling_std(x, N, W, std_out) == MLR_OK, "std with NaN OK");

    for (size_t i = W - 1; i < N; i++) {
        // Window [i-W+1, i] contains a bad value?
        int has_bad = 0;
        for (size_t j = i - W + 1; j <= i; j++) {
            if (!mlr_isfinite(x[j])) has_bad = 1;
        }

        if (has_bad) {
            ASSERT(mlr_isnan(mean_out[i]), "mean of window with bad value should be NAN");
            ASSERT(mlr_isnan(std_out[i]), "std of window with bad value should be NAN");
        } else {
            // Compare against naive recompute
            double mean = 0.0;
            for (size_t j = i - W + 1; j <= i; j++) mean += x[j];
            mean /= (double)W;
            double var = 0.0;
            for (size_t j = i - W + 1; j <= i; j++) {
                double d = x[j] - mean;
                var += d * d;
            }
            var /= (double)W;
            ASSERT(fabs(mean_out[i] - mean) < TOLERANCE, "mean should recover after bad value");
            ASSERT(fabs(std_out[i] - sqrt(var)) < TOLERANCE, "std should recover after bad value");
        }
    }

    // window == 1: NaN in, NaN out; finite in, 0 out
    ASSERT(mlr_rolling_std(x, N, 1, std_out) == MLR_OK, "window=1 with NaN OK");
    ASSERT(mlr_isnan(std_out[7]), "window=1 std of NaN should be NAN");
    ASSERT(std_out[8] == 0.0, "window=1 std of finite value should be 0");

    printf("  PASS: NaN recovery\n");
    return 0;
}

int test_rolling(void) {
    int failures = 0;
    failures += test_rolling_mean_basic();
    failures += test_rolling_mean_edge_cases();
    failures += test_rolling_std_basic();
    failures += test_rolling_sliding_consistency();
    failures += test_rolling_std_edge_cases();
    failures += test_rolling_shift_invariance();
    failures += test_rolling_nan_recovery();
    failures += test_ewma_vol_basic();
    failures += test_ewma_vol_regime_change();
    failures += test_ewma_vol_invalid_lambda();
    return failures;
}
