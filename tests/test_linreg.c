#include "mlrisk/linreg.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

#define TOLERANCE 1e-6

static int test_linreg_simple(void) {
    // Simple 1D case: y = 2*x + 1
    double X[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 3.0, 5.0, 7.0, 9.0};
    size_t n = 5;
    size_t d = 1;
    double ridge = 0.01;
    
    mlr_lin_model model;
    mlr_status status = mlr_lin_model_init(&model, d, ridge);
    ASSERT(status == MLR_OK, "mlr_lin_model_init should return MLR_OK");
    
    status = linreg_fit(X, y, n, d, ridge, &model);
    ASSERT(status == MLR_OK, "linreg_fit should return MLR_OK");
    
    // Check coefficients (should be close to w=2, b=1)
    ASSERT(fabs(model.w[0] - 2.0) < 0.1, "Weight should be close to 2.0");
    ASSERT(fabs(model.b - 1.0) < 0.1, "Bias should be close to 1.0");
    
    // Test prediction
    double X_test[] = {5.0, 6.0};
    double y_pred[2];
    status = linreg_predict(X_test, 2, d, &model, y_pred);
    ASSERT(status == MLR_OK, "linreg_predict should return MLR_OK");
    
    // y_pred[0] should be close to 2*5 + 1 = 11
    ASSERT(fabs(y_pred[0] - 11.0) < 0.5, "Prediction should be close to expected");
    
    mlr_lin_model_free(&model);
    
    printf("  PASS: linreg simple 1D case\n");
    return 0;
}

static int test_linreg_2d(void) {
    // 2D case: y = 1*x1 + 2*x2 + 3
    double X[] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 2.0
    };
    double y[] = {3.0, 4.0, 5.0, 6.0, 9.0};
    size_t n = 5;
    size_t d = 2;
    double ridge = 0.01;
    
    mlr_lin_model model;
    mlr_status status = mlr_lin_model_init(&model, d, ridge);
    ASSERT(status == MLR_OK, "mlr_lin_model_init should return MLR_OK");
    
    status = linreg_fit(X, y, n, d, ridge, &model);
    ASSERT(status == MLR_OK, "linreg_fit should return MLR_OK");
    
    // Check that predictions are reasonable
    double X_test[] = {1.0, 1.0};
    double y_pred[1];
    status = linreg_predict(X_test, 1, d, &model, y_pred);
    ASSERT(status == MLR_OK, "linreg_predict should return MLR_OK");
    
    // Should be close to 1*1 + 2*1 + 3 = 6
    ASSERT(fabs(y_pred[0] - 6.0) < 1.0, "Prediction should be reasonable");
    
    mlr_lin_model_free(&model);
    
    printf("  PASS: linreg 2D case\n");
    return 0;
}

static int test_linreg_invalid_inputs(void) {
    mlr_lin_model model;
    mlr_status status = mlr_lin_model_init(&model, 2, 0.01);
    ASSERT(status == MLR_OK, "mlr_lin_model_init should return MLR_OK");
    
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 2.0};
    
    // NULL inputs
    status = linreg_fit(NULL, y, 2, 2, 0.01, &model);
    ASSERT(status == MLR_EINVAL, "linreg_fit with NULL X should return MLR_EINVAL");
    
    status = linreg_fit(X, NULL, 2, 2, 0.01, &model);
    ASSERT(status == MLR_EINVAL, "linreg_fit with NULL y should return MLR_EINVAL");
    
    // Dimension mismatch
    status = linreg_fit(X, y, 2, 3, 0.01, &model);
    ASSERT(status == MLR_EINVAL, "linreg_fit with dimension mismatch should return MLR_EINVAL");
    
    double out[1];
    status = linreg_predict(NULL, 1, 2, &model, out);
    ASSERT(status == MLR_EINVAL, "linreg_predict with NULL X should return MLR_EINVAL");
    
    mlr_lin_model_free(&model);
    
    printf("  PASS: linreg invalid inputs\n");
    return 0;
}

static int test_lin_model_init_free(void) {
    mlr_lin_model model;

    mlr_status status = mlr_lin_model_init(&model, 5, 0.01);
    ASSERT(status == MLR_OK, "mlr_lin_model_init should return MLR_OK");
    ASSERT(model.d == 5, "model.d should be 5");
    ASSERT(model.ridge == 0.01, "model.ridge should be 0.01");
    ASSERT(model.w != NULL, "model.w should be allocated");
    ASSERT(model.b == 0.0, "model.b should be 0.0");

    mlr_lin_model_free(&model);
    ASSERT(model.w == NULL, "model.w should be NULL after free");
    ASSERT(model.d == 0, "model.d should be 0 after free");

    status = mlr_lin_model_init(NULL, 5, 0.01);
    ASSERT(status == MLR_EINVAL, "mlr_lin_model_init with NULL should return MLR_EINVAL");

    status = mlr_lin_model_init(&model, 0, 0.01);
    ASSERT(status == MLR_EINVAL, "mlr_lin_model_init with d=0 should return MLR_EINVAL");

    printf("  PASS: lin_model init/free\n");
    return 0;
}

static int test_linreg_uninitialized_model(void) {
    // Previously segfaulted: fit into a model that was never initialized
    double X[] = {0.0, 1.0, 2.0};
    double y[] = {1.0, 3.0, 5.0};

    mlr_lin_model model;
    model.d = 1;
    model.w = NULL;

    mlr_status status = linreg_fit(X, y, 3, 1, 0.01, &model);
    ASSERT(status == MLR_EINVAL, "linreg_fit on uninitialized model should return MLR_EINVAL");

    printf("  PASS: linreg uninitialized model rejected\n");
    return 0;
}

static int test_linreg_scale_invariance(void) {
    // y = 2e8 * x + 1 with tiny-scale features; the old absolute 1e-10
    // singularity threshold falsely rejected this system
    double X[] = {0.0, 1e-8, 2e-8, 3e-8, 4e-8};
    double y[] = {1.0, 3.0, 5.0, 7.0, 9.0};

    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 1, 0.0) == MLR_OK, "init should succeed");

    mlr_status status = linreg_fit(X, y, 5, 1, 0.0, &model);
    ASSERT(status == MLR_OK, "linreg_fit on small-scale features should return MLR_OK");
    ASSERT(fabs(model.w[0] - 2e8) / 2e8 < 1e-6, "slope should be recovered at small scale");

    mlr_lin_model_free(&model);
    printf("  PASS: linreg scale invariance\n");
    return 0;
}

static int test_linreg_singular_matrix(void) {
    // Duplicate columns with no ridge: genuinely singular
    double X[] = {
        1.0, 1.0,
        2.0, 2.0,
        3.0, 3.0,
        4.0, 4.0
    };
    double y[] = {1.0, 2.0, 3.0, 4.0};

    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 2, 0.0) == MLR_OK, "init should succeed");

    mlr_status status = linreg_fit(X, y, 4, 2, 0.0, &model);
    ASSERT(status == MLR_EDOMAIN, "linreg_fit on singular matrix should return MLR_EDOMAIN");

    mlr_lin_model_free(&model);
    printf("  PASS: linreg singular matrix detected\n");
    return 0;
}

int test_linreg(void) {
    int failures = 0;
    failures += test_linreg_simple();
    failures += test_linreg_2d();
    failures += test_linreg_invalid_inputs();
    failures += test_lin_model_init_free();
    failures += test_linreg_uninitialized_model();
    failures += test_linreg_scale_invariance();
    failures += test_linreg_singular_matrix();
    return failures;
}
