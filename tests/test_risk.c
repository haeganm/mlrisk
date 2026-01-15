#include "mlrisk/risk.h"
#include "mlrisk/rolling.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

#define TOLERANCE 1e-9

static int test_risk_forecast_ewma(void) {
    double returns[] = {0.01, -0.02, 0.015, -0.01, 0.02};
    double sigma_out[5];
    double lambda = 0.94;
    
    mlr_status status = risk_forecast_ewma(returns, 5, lambda, sigma_out);
    ASSERT(status == MLR_OK, "risk_forecast_ewma should return MLR_OK");
    
    // Should match ewma_vol directly
    double direct_out[5];
    status = ewma_vol(returns, 5, lambda, direct_out);
    ASSERT(status == MLR_OK, "ewma_vol should return MLR_OK");
    
    for (size_t i = 0; i < 5; i++) {
        ASSERT(fabs(sigma_out[i] - direct_out[i]) < TOLERANCE,
               "risk_forecast_ewma should match ewma_vol");
    }
    
    printf("  PASS: risk_forecast_ewma matches ewma_vol\n");
    return 0;
}

static int test_lin_model_init_free(void) {
    mlr_lin_model model;
    
    // Test initialization
    mlr_status status = mlr_lin_model_init(&model, 5, 0.01);
    ASSERT(status == MLR_OK, "mlr_lin_model_init should return MLR_OK");
    ASSERT(model.d == 5, "model.d should be 5");
    ASSERT(model.ridge == 0.01, "model.ridge should be 0.01");
    ASSERT(model.w != NULL, "model.w should be allocated");
    ASSERT(model.b == 0.0, "model.b should be 0.0");
    
    // Test free
    mlr_lin_model_free(&model);
    ASSERT(model.w == NULL, "model.w should be NULL after free");
    ASSERT(model.d == 0, "model.d should be 0 after free");
    
    printf("  PASS: lin_model init/free\n");
    return 0;
}

static int test_lin_model_init_invalid(void) {
    mlr_lin_model model;
    
    // NULL model
    mlr_status status = mlr_lin_model_init(NULL, 5, 0.01);
    ASSERT(status == MLR_EINVAL, "mlr_lin_model_init with NULL should return MLR_EINVAL");
    
    // Zero dimension
    status = mlr_lin_model_init(&model, 0, 0.01);
    ASSERT(status == MLR_EINVAL, "mlr_lin_model_init with d=0 should return MLR_EINVAL");
    
    printf("  PASS: lin_model init invalid inputs\n");
    return 0;
}

int test_risk(void) {
    int failures = 0;
    failures += test_risk_forecast_ewma();
    failures += test_lin_model_init_free();
    failures += test_lin_model_init_invalid();
    return failures;
}
