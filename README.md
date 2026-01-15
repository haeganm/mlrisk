# mlrisk - Risk Forecasting and Position Sizing Library

A C11 static library for volatility forecasting and position sizing with time-series-safe evaluation utilities.

## Overview

mlrisk provides:

- **Rolling Statistics**: Rolling mean, standard deviation, and EWMA (Exponentially Weighted Moving Average) volatility
- **Volatility Forecasting**: EWMA-based volatility forecasting with optional linear model support
- **Position Sizing**: Volatility targeting and risk capping with leverage constraints
- **Time-Series Utilities**: Walk-forward split utilities for backtesting without lookahead bias
- **Linear Regression**: Optional closed-form ridge regression for volatility modeling

## Features

- Pure C11, no external dependencies
- Numerically stable algorithms (Welford's method for rolling stats)
- Comprehensive input validation and error handling
- Time-series safe (no lookahead bias in utilities)
- Well-tested with unit tests
- Cross-platform (macOS, Linux)

## Building

### Prerequisites

- CMake 3.15 or higher
- C11-compatible compiler (GCC, Clang, MSVC)

### Build Instructions

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Running Tests

```bash
ctest
```

Or run the test executable directly:

```bash
./mlrisk_tests
```

### Running the Demo

```bash
./vol_target_demo
```

## Module Overview

### Core Types (`include/mlrisk/types.h`)

- Error codes: `MLR_OK`, `MLR_EINVAL`, `MLR_ENOMEM`, `MLR_EBOUNDS`, `MLR_EDOMAIN`
- Utility functions: `mlr_isnan()`, `mlr_isfinite()`
- Constants: `MLR_NAN`

### Rolling Statistics (`include/mlrisk/rolling.h`)

- `rolling_mean()`: Compute rolling mean of a time series
- `rolling_std()`: Compute rolling standard deviation (numerically stable)
- `ewma_vol()`: Compute EWMA volatility (per-period sigma)

### Risk Forecasting (`include/mlrisk/risk.h`)

- `risk_forecast_ewma()`: EWMA-based volatility forecast
- `mlr_lin_model`: Structure for linear volatility models
- `mlr_lin_model_init()`, `mlr_lin_model_free()`: Model memory management

### Position Sizing (`include/mlrisk/sizing.h`)

- `vol_target_position()`: Compute position sizes using volatility targeting
- `risk_cap_position()`: Compute position sizes using risk capping

**Important**: All volatility values (sigma, target_vol) are **per-period**. To convert annualized volatility to per-period:

```
per_period_vol = annualized_vol / sqrt(periods_per_year)
```

For daily data: `per_period_vol = annualized_vol / sqrt(252)`

### Walk-Forward Splits (`include/mlrisk/split.h`)

- `walk_forward_ranges()`: Generate train/test splits for time-series cross-validation
- Ensures no lookahead: test data always comes after training data

### Linear Regression (`include/mlrisk/linreg.h`) - Optional

- `linreg_fit()`: Fit ridge regression model
- `linreg_predict()`: Predict using fitted model
- Uses Gaussian elimination with partial pivoting for small feature dimensions

## Example Usage

```c
#include "mlrisk/mlrisk.h"
#include <stdio.h>

int main(void) {
    // Example: Compute EWMA volatility and position sizing
    double returns[] = {0.01, -0.02, 0.015, -0.01, 0.02};
    double prices[] = {100.0, 99.0, 100.5, 99.5, 101.5};
    size_t n = 5;
    
    // Compute EWMA volatility
    double sigma[n];
    mlr_status status = risk_forecast_ewma(returns, n, 0.94, sigma);
    if (status != MLR_OK) {
        fprintf(stderr, "Error computing volatility\n");
        return 1;
    }
    
    // Compute position sizes
    double positions[n];
    status = vol_target_position(
        sigma,           // volatility forecast
        0.01,            // target volatility (1% per-period)
        100000.0,        // account equity
        prices,          // asset prices
        2.0,             // max leverage
        n,
        positions
    );
    
    if (status != MLR_OK) {
        fprintf(stderr, "Error computing positions\n");
        return 1;
    }
    
    // Print results
    for (size_t i = 0; i < n; i++) {
        printf("Period %zu: sigma=%.6f, position=%.2f\n",
               i, sigma[i], positions[i]);
    }
    
    return 0;
}
```

## Volatility Conventions

### Per-Period vs Annualized

mlrisk uses **per-period volatility** throughout:

- Input `sigma` values should be per-period
- Input `target_vol` should be per-period
- Output `sigma` values are per-period

**Conversion examples:**

- Daily data: `daily_vol = annualized_vol / sqrt(252)`
- Weekly data: `weekly_vol = annualized_vol / sqrt(52)`
- Monthly data: `monthly_vol = annualized_vol / sqrt(12)`

### EWMA Parameters

Typical `lambda` values for EWMA:
- Daily data: 0.94 - 0.97
- Higher lambda = slower decay = more persistent volatility

## Lookahead Avoidance

The walk-forward split utilities (`walk_forward_ranges`) are designed to prevent lookahead bias:

- Training data always comes before test data
- No overlap between training and test sets
- Test start index >= training end index

When using these utilities for backtesting:
1. Train your model on `[train_start, train_end)`
2. Test on `[test_start, test_end)` where `test_start >= train_end`
3. Never use future data to predict past values

## API Stability

The v1 API is designed to be minimal and stable. Future versions may add:
- Additional forecasting models
- Portfolio-level position sizing
- More sophisticated risk metrics

## License

[Specify your license here]

## Contributing

[Contributing guidelines if applicable]
