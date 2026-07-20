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

static int test_kelly_fraction(void) {
    // mean = 0.025, sample var = 0.0075 -> full Kelly 10/3, half Kelly 5/3
    double returns[] = {0.1, -0.05, 0.1, -0.05};
    double f = 0.0;

    mlr_status status = mlr_kelly_fraction(returns, 4, 1.0, &f);
    ASSERT(status == MLR_OK, "kelly should return MLR_OK");
    ASSERT(fabs(f - 10.0 / 3.0) < TOLERANCE, "full Kelly should be 10/3");

    status = mlr_kelly_fraction(returns, 4, 0.5, &f);
    ASSERT(status == MLR_OK, "half kelly should return MLR_OK");
    ASSERT(fabs(f - 5.0 / 3.0) < TOLERANCE, "half Kelly should be 5/3");

    // Negative edge -> negative fraction
    double losing[] = {-0.1, 0.05, -0.1, 0.05};
    status = mlr_kelly_fraction(losing, 4, 1.0, &f);
    ASSERT(status == MLR_OK, "negative-edge kelly should return MLR_OK");
    ASSERT(f < 0.0, "negative edge should give negative Kelly fraction");

    // Degenerate inputs
    double constant[] = {0.01, 0.01, 0.01};
    ASSERT(mlr_kelly_fraction(constant, 3, 1.0, &f) == MLR_EDOMAIN,
           "zero variance should return MLR_EDOMAIN");
    ASSERT(mlr_kelly_fraction(returns, 1, 1.0, &f) == MLR_EINVAL, "n<2 -> MLR_EINVAL");
    ASSERT(mlr_kelly_fraction(returns, 4, 0.0, &f) == MLR_EINVAL, "fraction=0 -> MLR_EINVAL");
    ASSERT(mlr_kelly_fraction(NULL, 4, 1.0, &f) == MLR_EINVAL, "NULL returns -> MLR_EINVAL");

    printf("  PASS: kelly fraction\n");
    return 0;
}

static int test_drawdown_scale(void) {
    // Peaks 100,110,110,110,120; drawdowns 0,0,0.1,0.05,0
    double equity[] = {100.0, 110.0, 99.0, 104.5, 120.0};
    double scale[5];

    mlr_status status = mlr_drawdown_scale(equity, 5, 0.2, scale);
    ASSERT(status == MLR_OK, "drawdown_scale should return MLR_OK");
    ASSERT(fabs(scale[0] - 1.0) < TOLERANCE, "scale[0] should be 1");
    ASSERT(fabs(scale[1] - 1.0) < TOLERANCE, "scale[1] should be 1");
    ASSERT(fabs(scale[2] - 0.5) < TOLERANCE, "scale[2] should be 0.5 (10% dd of 20% cap)");
    ASSERT(fabs(scale[3] - 0.75) < TOLERANCE, "scale[3] should be 0.75");
    ASSERT(fabs(scale[4] - 1.0) < TOLERANCE, "scale[4] should be 1 (new peak)");

    // Drawdown beyond max_dd clamps to zero exposure
    double crash[] = {100.0, 50.0};
    status = mlr_drawdown_scale(crash, 2, 0.2, scale);
    ASSERT(status == MLR_OK, "crash path should return MLR_OK");
    ASSERT(scale[1] == 0.0, "drawdown past max_dd should scale to 0");

    // Invalid parameters
    ASSERT(mlr_drawdown_scale(equity, 5, 0.0, scale) == MLR_EINVAL, "max_dd=0 -> MLR_EINVAL");
    ASSERT(mlr_drawdown_scale(equity, 5, 1.5, scale) == MLR_EINVAL, "max_dd>1 -> MLR_EINVAL");
    double bad[] = {100.0, 0.0};
    ASSERT(mlr_drawdown_scale(bad, 2, 0.2, scale) == MLR_EDOMAIN,
           "non-positive equity -> MLR_EDOMAIN");

    printf("  PASS: drawdown scale\n");
    return 0;
}

int test_sizing(void) {
    int failures = 0;
    failures += test_kelly_fraction();
    failures += test_drawdown_scale();
    failures += test_vol_target_position_basic();
    failures += test_vol_target_position_nan_sigma();
    failures += test_vol_target_position_leverage_cap();
    failures += test_vol_target_position_decreasing_with_sigma();
    failures += test_vol_target_position_as_risk_cap();
    failures += test_sizing_invalid_inputs();
    return failures;
}
