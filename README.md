# mlrisk

> A lightweight C11 library for volatility forecasting and position sizing with time-series-safe evaluation utilities.

[![CI](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml/badge.svg)](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml)
[![C11](https://img.shields.io/badge/C-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

## Overview

**mlrisk** is a C11 static library for quantitative finance, focused on correctness and portability:

- **Volatility forecasting** — EWMA-based estimation with optional linear regression models
- **Position sizing** — volatility targeting and risk capping with leverage constraints
- **Time-series utilities** — walk-forward splits for backtesting without lookahead bias
- **Rolling statistics** — numerically stable rolling mean, standard deviation, and EWMA (Welford's method)

Pure C11 with zero external dependencies, cross-platform (Windows, macOS, Linux), with full input validation and a unit test suite run in CI on all three platforms.

**Disclaimer:** This project is for educational and research purposes only and does not constitute financial, investment, or trading advice. Trading involves risk, including the possible loss of principal. Use at your own risk.

## Quick Start

Prerequisites: [CMake](https://cmake.org/download/) 3.15+ and any C11 compiler (GCC 4.9+, Clang 3.3+, MSVC 2015+).

```bash
git clone https://github.com/haeganm/mlrisk.git
cd mlrisk

# Configure and build
cmake -S . -B build
cmake --build build --config Release

# Run tests
ctest --test-dir build -C Release --output-on-failure

# Try the demo
./build/Release/vol_target_demo    # Linux/macOS
.\build\Release\vol_target_demo.exe  # Windows
```

<details><summary>Platform-specific build variants</summary>

```bash
# Linux / macOS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Windows (Visual Studio)
cmake -S . -B build
cmake --build build --config Release

# Windows (MinGW)
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
</details>

## Usage

### Basic Example: Volatility Forecasting & Position Sizing

```c
#include "mlrisk/mlrisk.h"
#include <stdio.h>

int main(void) {
    // Sample returns data
    double returns[] = {0.01, -0.02, 0.015, -0.01, 0.02};
    double prices[] = {100.0, 99.0, 100.5, 99.5, 101.5};
    size_t n = 5;

    // 1. Compute EWMA volatility
    double sigma[5];
    mlr_status status = risk_forecast_ewma(returns, n, 0.94, sigma);
    if (status != MLR_OK) {
        fprintf(stderr, "Error computing volatility\n");
        return 1;
    }

    // 2. Compute position sizes using volatility targeting
    double positions[5];
    status = vol_target_position(
        sigma,           // volatility forecast
        0.01,            // target volatility (1% per-period)
        100000.0,        // account equity ($100k)
        prices,          // asset prices
        2.0,             // max leverage (2x)
        n,
        positions
    );

    if (status != MLR_OK) {
        fprintf(stderr, "Error computing positions\n");
        return 1;
    }

    // 3. Print results
    for (size_t i = 0; i < n; i++) {
        printf("Period %zu: sigma=%.6f, position=%.2f shares\n",
               i, sigma[i], positions[i]);
    }

    return 0;
}
```

### Linking in Your Project

Add to your `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/mlrisk)

target_link_libraries(your_target PRIVATE mlrisk)
target_include_directories(your_target PRIVATE path/to/mlrisk/include)
```

## API Reference

### Rolling Statistics (`include/mlrisk/rolling.h`)

```c
// Compute rolling mean
mlr_status rolling_mean(const double *x, size_t n, size_t window, double *out);

// Compute rolling standard deviation (numerically stable)
mlr_status rolling_std(const double *x, size_t n, size_t window, double *out);

// Compute EWMA volatility (per-period sigma)
mlr_status ewma_vol(const double *returns, size_t n, double lambda, double *out);
```

### Risk Forecasting (`include/mlrisk/risk.h`)

```c
// EWMA-based volatility forecast
mlr_status risk_forecast_ewma(const double *returns, size_t n, double lambda, double *sigma_out);

// Linear model for volatility forecasting
mlr_lin_model model;
mlr_lin_model_init(&model, feature_dim, ridge_param);
// ... use model ...
mlr_lin_model_free(&model);
```

### Position Sizing (`include/mlrisk/sizing.h`)

```c
// Volatility targeting
mlr_status vol_target_position(
    const double *sigma,      // per-period volatility
    double target_vol,        // target per-period volatility
    double equity,            // account equity
    const double *price,      // asset prices
    double max_leverage,      // maximum leverage multiplier
    size_t n,
    double *position_out
);

// Risk capping
mlr_status risk_cap_position(
    const double *sigma,
    double risk_cap,          // max risk per position (fraction of equity)
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
);
```

### Walk-Forward Splits (`include/mlrisk/split.h`)

```c
// Generate train/test splits for time-series cross-validation
mlr_range ranges[100];
size_t count = walk_forward_ranges(
    n,              // total data length
    train_len,      // training window size
    test_len,       // test window size
    step,           // step size between splits
    ranges          // output array
);
```

### Linear Regression (`include/mlrisk/linreg.h`)

```c
// Fit ridge regression model
mlr_status linreg_fit(
    const double *X,      // feature matrix (n rows, d columns)
    const double *y,      // target vector
    size_t n,             // number of samples
    size_t d,             // number of features
    double ridge,         // ridge regularization parameter
    mlr_lin_model *model_out
);

// Predict using fitted model
mlr_status linreg_predict(
    const double *X,
    size_t n,
    size_t d,
    const mlr_lin_model *model,
    double *out
);
```

### Error Handling

All functions return `mlr_status`:

- `MLR_OK` - Success
- `MLR_EINVAL` - Invalid argument
- `MLR_ENOMEM` - Memory allocation failure
- `MLR_EBOUNDS` - Index out of bounds
- `MLR_EDOMAIN` - Domain error

## Conventions

### Volatility

mlrisk uses **per-period volatility** throughout. To convert annualized volatility to per-period:

```
per_period_vol = annualized_vol / sqrt(periods_per_year)
```

- Daily data: `daily_vol = annualized_vol / sqrt(252)`
- Weekly data: `weekly_vol = annualized_vol / sqrt(52)`
- Monthly data: `monthly_vol = annualized_vol / sqrt(12)`

### EWMA Parameters

Typical `lambda` values for daily data: 0.94 - 0.97. Higher lambda = slower decay = more persistent volatility.

### Lookahead Avoidance

`walk_forward_ranges()` guarantees training data always precedes test data with no overlap (`test_start >= train_end` for all ranges). When backtesting, always use walk-forward splits to avoid overfitting and unrealistic performance estimates.

## License

MIT - see [LICENSE](LICENSE).
