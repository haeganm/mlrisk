#include "mlrisk/linreg.h"
#include "mlrisk/risk.h"
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

int test_linreg(void) {
    int failures = 0;
    failures += test_linreg_simple();
    failures += test_linreg_2d();
    failures += test_linreg_invalid_inputs();
    return failures;
}
