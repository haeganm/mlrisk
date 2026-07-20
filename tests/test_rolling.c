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
    
    mlr_status status = rolling_mean(x, 5, 3, out);
    ASSERT(status == MLR_OK, "rolling_mean should return MLR_OK");
    
    // First two should be NAN
    ASSERT(mlr_isnan(out[0]), "out[0] should be NAN");
    ASSERT(mlr_isnan(out[1]), "out[1] should be NAN");
    
    // out[2] = mean of [1,2,3] = 2.0
    ASSERT(fabs(out[2] - 2.0) < TOLERANCE, "out[2] should be 2.0");
    
    // out[3] = mean of [2,3,4] = 3.0
    ASSERT(fabs(out[3] - 3.0) < TOLERANCE, "out[3] should be 3.0");
    
    // out[4] = mean of [3,4,5] = 4.0
    ASSERT(fabs(out[4] - 4.0) < TOLERANCE, "out[4] should be 4.0");
    
    printf("  PASS: rolling_mean basic\n");
    return 0;
}

static int test_rolling_mean_edge_cases(void) {
    double x[] = {1.0};
    double out[1];
    
    // Window = 1 should work
    mlr_status status = rolling_mean(x, 1, 1, out);
    ASSERT(status == MLR_OK, "rolling_mean with window=1 should work");
    ASSERT(fabs(out[0] - 1.0) < TOLERANCE, "out[0] should be 1.0");
    
    // Window > n should set all to NAN
    status = rolling_mean(x, 1, 5, out);
    ASSERT(status == MLR_OK, "rolling_mean with window>n should return OK");
    ASSERT(mlr_isnan(out[0]), "out[0] should be NAN when window>n");
    
    // NULL pointers
    status = rolling_mean(NULL, 1, 1, out);
    ASSERT(status == MLR_EINVAL, "rolling_mean with NULL x should return MLR_EINVAL");
    
    status = rolling_mean(x, 1, 1, NULL);
    ASSERT(status == MLR_EINVAL, "rolling_mean with NULL out should return MLR_EINVAL");
    
    // Zero window
    status = rolling_mean(x, 1, 0, out);
    ASSERT(status == MLR_EINVAL, "rolling_mean with window=0 should return MLR_EINVAL");
    
    printf("  PASS: rolling_mean edge cases\n");
    return 0;
}

static int test_rolling_std_basic(void) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5];
    
    mlr_status status = rolling_std(x, 5, 3, out);
    ASSERT(status == MLR_OK, "rolling_std should return MLR_OK");
    
    // First two should be NAN
    ASSERT(mlr_isnan(out[0]), "out[0] should be NAN");
    ASSERT(mlr_isnan(out[1]), "out[1] should be NAN");
    
    // out[2] = std of [1,2,3]
    // mean = 2.0, variance = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
    // std = sqrt(2/3) ≈ 0.8165
    double expected = sqrt(2.0 / 3.0);
    ASSERT(fabs(out[2] - expected) < TOLERANCE, "out[2] should match expected std");
    
    printf("  PASS: rolling_std basic\n");
    return 0;
}

static int test_ewma_vol_basic(void) {
    double returns[] = {0.01, -0.02, 0.015, -0.01, 0.02};
    double out[5];
    double lambda = 0.94;
    
    mlr_status status = ewma_vol(returns, 5, lambda, out);
    ASSERT(status == MLR_OK, "ewma_vol should return MLR_OK");
    
    // First value should be |returns[0]|
    ASSERT(fabs(out[0] - fabs(returns[0])) < TOLERANCE, "out[0] should be |returns[0]|");
    
    // Subsequent values should be positive
    for (size_t i = 1; i < 5; i++) {
        ASSERT(out[i] > 0.0, "EWMA vol should be positive");
    }
    
    printf("  PASS: ewma_vol basic\n");
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
    
    mlr_status status = ewma_vol(returns, 100, lambda, out);
    ASSERT(status == MLR_OK, "ewma_vol should return MLR_OK");
    
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
    
    printf("  PASS: ewma_vol regime change\n");
    return 0;
}

static int test_ewma_vol_invalid_lambda(void) {
    double returns[] = {0.01};
    double out[1];
    
    mlr_status status = ewma_vol(returns, 1, -0.1, out);
    ASSERT(status == MLR_EINVAL, "ewma_vol with negative lambda should return MLR_EINVAL");
    
    status = ewma_vol(returns, 1, 1.5, out);
    ASSERT(status == MLR_EINVAL, "ewma_vol with lambda>1 should return MLR_EINVAL");
    
    printf("  PASS: ewma_vol invalid lambda\n");
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
        ASSERT(rolling_mean(x, N, w, fast_mean) == MLR_OK, "sliding mean OK");
        ASSERT(rolling_std(x, N, w, fast_std) == MLR_OK, "sliding std OK");

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
    ASSERT(rolling_std(constant, 10, 4, out) == MLR_OK, "constant std OK");
    for (size_t i = 3; i < 10; i++) {
        ASSERT(out[i] == 0.0, "std of constant array should be exactly 0");
    }

    // window > n: all NAN
    ASSERT(rolling_std(constant, 3, 5, out) == MLR_OK, "window>n std OK");
    for (size_t i = 0; i < 3; i++) {
        ASSERT(mlr_isnan(out[i]), "std should be NAN when window>n");
    }

    // window == 1: zeros
    ASSERT(rolling_std(constant, 3, 1, out) == MLR_OK, "window=1 std OK");
    ASSERT(out[0] == 0.0 && out[2] == 0.0, "std with window=1 should be 0");

    printf("  PASS: rolling_std edge cases\n");
    return 0;
}

int test_rolling(void) {
    int failures = 0;
    failures += test_rolling_mean_basic();
    failures += test_rolling_mean_edge_cases();
    failures += test_rolling_std_basic();
    failures += test_rolling_sliding_consistency();
    failures += test_rolling_std_edge_cases();
    failures += test_ewma_vol_basic();
    failures += test_ewma_vol_regime_change();
    failures += test_ewma_vol_invalid_lambda();
    return failures;
}
