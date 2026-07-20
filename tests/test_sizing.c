#include "mlrisk/sizing.h"
#include <stdio.h>
#include <math.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

#define TOLERANCE 1e-9

static int test_vol_target_position_basic(void) {
    double sigma[] = {0.01, 0.02, 0.015};
    double price[] = {100.0, 100.0, 100.0};
    double position_out[3];
    double target_vol = 0.01;
    double equity = 10000.0;
    double max_leverage = 2.0;
    
    mlr_status status = mlr_vol_target_position(
        sigma, target_vol, equity, price, max_leverage, 3, position_out
    );
    ASSERT(status == MLR_OK, "mlr_vol_target_position should return MLR_OK");
    
    // position[0] = (0.01/0.01) * (10000/100) = 100.0
    ASSERT(fabs(position_out[0] - 100.0) < TOLERANCE, "position[0] should be 100.0");
    
    // position[1] = (0.01/0.02) * (10000/100) = 50.0
    ASSERT(fabs(position_out[1] - 50.0) < TOLERANCE, "position[1] should be 50.0");
    
    // position[2] = (0.01/0.015) * (10000/100) ≈ 66.67
    ASSERT(fabs(position_out[2] - (10000.0 / 150.0)) < TOLERANCE,
           "position[2] should match expected");
    
    printf("  PASS: mlr_vol_target_position basic\n");
    return 0;
}

static int test_vol_target_position_nan_sigma(void) {
    double sigma[] = {0.01, MLR_NAN, 0.015, -0.01};
    double price[] = {100.0, 100.0, 100.0, 100.0};
    double position_out[4];
    double target_vol = 0.01;
    double equity = 10000.0;
    double max_leverage = 2.0;
    
    mlr_status status = mlr_vol_target_position(
        sigma, target_vol, equity, price, max_leverage, 4, position_out
    );
    ASSERT(status == MLR_OK, "mlr_vol_target_position should return MLR_OK");
    
    ASSERT(fabs(position_out[0] - 100.0) < TOLERANCE, "position[0] should be 100.0");
    ASSERT(fabs(position_out[1] - 0.0) < TOLERANCE, "position[1] should be 0.0 (NAN sigma)");
    ASSERT(fabs(position_out[2] - (10000.0 / 150.0)) < TOLERANCE, "position[2] should be valid");
    ASSERT(fabs(position_out[3] - 0.0) < TOLERANCE, "position[3] should be 0.0 (negative sigma)");
    
    printf("  PASS: mlr_vol_target_position NAN/negative sigma\n");
    return 0;
}

static int test_vol_target_position_leverage_cap(void) {
    double sigma[] = {0.001}; // Very low sigma -> very high position
    double price[] = {100.0};
    double position_out[1];
    double target_vol = 0.01;
    double equity = 10000.0;
    double max_leverage = 2.0;
    
    mlr_status status = mlr_vol_target_position(
        sigma, target_vol, equity, price, max_leverage, 1, position_out
    );
    ASSERT(status == MLR_OK, "mlr_vol_target_position should return MLR_OK");
    
    // Without cap: position = (0.01/0.001) * (10000/100) = 1000.0
    // With cap: max_notional = 2.0 * 10000 = 20000, so position = 20000/100 = 200.0
    ASSERT(fabs(position_out[0] - 200.0) < TOLERANCE,
           "position should be capped by leverage");
    
    printf("  PASS: mlr_vol_target_position leverage cap\n");
    return 0;
}

static int test_vol_target_position_decreasing_with_sigma(void) {
    double sigma[] = {0.01, 0.02, 0.03, 0.04};
    double price[] = {100.0, 100.0, 100.0, 100.0};
    double position_out[4];
    double target_vol = 0.01;
    double equity = 10000.0;
    double max_leverage = 10.0; // High leverage to avoid capping
    
    mlr_status status = mlr_vol_target_position(
        sigma, target_vol, equity, price, max_leverage, 4, position_out
    );
    ASSERT(status == MLR_OK, "mlr_vol_target_position should return MLR_OK");
    
    // Position should decrease as sigma increases
    for (size_t i = 1; i < 4; i++) {
        ASSERT(position_out[i] < position_out[i-1],
               "position should decrease as sigma increases");
    }
    
    printf("  PASS: mlr_vol_target_position decreasing with sigma\n");
    return 0;
}

static int test_vol_target_position_as_risk_cap(void) {
    // Risk capping is the same formula with target_vol read as the risk cap
    double sigma[] = {0.01, 0.02};
    double price[] = {100.0, 100.0};
    double position_out[2];
    double risk_cap = 0.02; // 2% risk per position
    double equity = 10000.0;
    double max_leverage = 2.0;

    mlr_status status = mlr_vol_target_position(
        sigma, risk_cap, equity, price, max_leverage, 2, position_out
    );
    ASSERT(status == MLR_OK, "mlr_vol_target_position should return MLR_OK");

    // position[0] = (0.02/0.01) * (10000/100) = 200.0
    ASSERT(fabs(position_out[0] - 200.0) < TOLERANCE, "position[0] should be 200.0");

    // position[1] = (0.02/0.02) * (10000/100) = 100.0
    ASSERT(fabs(position_out[1] - 100.0) < TOLERANCE, "position[1] should be 100.0");

    printf("  PASS: mlr_vol_target_position as risk cap\n");
    return 0;
}

static int test_sizing_invalid_inputs(void) {
    double sigma[] = {0.01};
    double price[] = {100.0};
    double position_out[1];
    
    // NULL pointers
    mlr_status status = mlr_vol_target_position(
        NULL, 0.01, 10000.0, price, 2.0, 1, position_out
    );
    ASSERT(status == MLR_EINVAL, "mlr_vol_target_position with NULL sigma should return MLR_EINVAL");
    
    status = mlr_vol_target_position(
        sigma, 0.01, 10000.0, NULL, 2.0, 1, position_out
    );
    ASSERT(status == MLR_EINVAL, "mlr_vol_target_position with NULL price should return MLR_EINVAL");
    
    // Invalid equity or leverage
    status = mlr_vol_target_position(
        sigma, 0.01, -1000.0, price, 2.0, 1, position_out
    );
    ASSERT(status == MLR_EINVAL, "mlr_vol_target_position with negative equity should return MLR_EINVAL");
    
    status = mlr_vol_target_position(
        sigma, 0.01, 10000.0, price, -1.0, 1, position_out
    );
    ASSERT(status == MLR_EINVAL, "mlr_vol_target_position with negative leverage should return MLR_EINVAL");

    // Invalid target_vol (previously produced uncapped short positions)
    status = mlr_vol_target_position(
        sigma, -0.01, 10000.0, price, 2.0, 1, position_out
    );
    ASSERT(status == MLR_EINVAL, "mlr_vol_target_position with negative target_vol should return MLR_EINVAL");

    status = mlr_vol_target_position(
        sigma, 0.0, 10000.0, price, 2.0, 1, position_out
    );
    ASSERT(status == MLR_EINVAL, "mlr_vol_target_position with zero target_vol should return MLR_EINVAL");

    status = mlr_vol_target_position(
        sigma, MLR_NAN, 10000.0, price, 2.0, 1, position_out
    );
    ASSERT(status == MLR_EINVAL, "mlr_vol_target_position with NaN target_vol should return MLR_EINVAL");

    printf("  PASS: sizing invalid inputs\n");
    return 0;
}

int test_sizing(void) {
    int failures = 0;
    failures += test_vol_target_position_basic();
    failures += test_vol_target_position_nan_sigma();
    failures += test_vol_target_position_leverage_cap();
    failures += test_vol_target_position_decreasing_with_sigma();
    failures += test_vol_target_position_as_risk_cap();
    failures += test_sizing_invalid_inputs();
    return failures;
}
