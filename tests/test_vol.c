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

int test_vol(void) {
    int failures = 0;
    failures += test_parkinson_known_answer();
    failures += test_garman_klass_known_answer();
    return failures;
}
