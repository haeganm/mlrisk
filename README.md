# mlrisk

> A lightweight C11 library for volatility forecasting and position sizing with time-series-safe evaluation utilities.

[![CI](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml/badge.svg)](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml)
[![C11](https://img.shields.io/badge/C-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

## Overview

**mlrisk** is a C11 static library for quantitative finance, focused on correctness and portability:

- **Volatility estimators** — EWMA, GARCH(1,1) fitted by MLE, and range-based Parkinson / Garman-Klass
- **Position sizing** — volatility targeting, Kelly fractions, and drawdown-based exposure scaling
- **Purged + embargoed walk-forward splits** — backtest without lookahead or label-overlap leakage
- **Rolling statistics** — O(n) sliding mean and standard deviation (rolling Welford updates)
- **Ridge regression** — closed-form linear models for volatility forecasting

Pure C11 with zero external dependencies, cross-platform (Windows, macOS, Linux), built with `-Werror`, and tested in CI on all three platforms plus an AddressSanitizer/UBSan job.

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

### Volatility Forecasting & Position Sizing

```c
#include "mlrisk/mlrisk.h"
#include <stdio.h>

int main(void) {
    double returns[] = {0.01, -0.02, 0.015, -0.01, 0.02};
    double prices[] = {100.0, 99.0, 100.5, 99.5, 101.5};
    size_t n = 5;

    // 1. Compute EWMA volatility
    double sigma[5];
    mlr_status status = mlr_ewma_vol(returns, n, 0.94, sigma);
    if (status != MLR_OK) {
        fprintf(stderr, "Error computing volatility\n");
        return 1;
    }

    // 2. Compute position sizes using volatility targeting
    double positions[5];
    status = mlr_vol_target_position(
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

    for (size_t i = 0; i < n; i++) {
        printf("Period %zu: sigma=%.6f, position=%.2f shares\n",
               i, sigma[i], positions[i]);
    }
    return 0;
}
```

### Purged Walk-Forward Splits

```c
#include "mlrisk/split.h"
#include <stdlib.h>

// Count-query first, then fill - no guessing at array sizes
size_t count;
mlr_walk_forward_splits(n, 252, 21, 21, 5, 5, 0, NULL, 0, &count);

mlr_split *splits = malloc(count * sizeof *splits);
mlr_walk_forward_splits(n, 252, 21, 21, 5, 5, 0, splits, count, &count);

for (size_t i = 0; i < count; i++) {
    // train on [splits[i].train_start, splits[i].train_end)
    // evaluate on [splits[i].test_start, splits[i].test_end)
}
free(splits);
```

### Linking in Your Project

Via `add_subdirectory`:

```cmake
add_subdirectory(path/to/mlrisk)
target_link_libraries(your_target PRIVATE mlrisk::mlrisk)
```

Or install it (`cmake --install build`) and use `find_package`:

```cmake
find_package(mlrisk 2 REQUIRED)
target_link_libraries(your_target PRIVATE mlrisk::mlrisk)
```

## API Reference

### Volatility Estimators (`include/mlrisk/vol.h`, `rolling.h`)

```c
// EWMA volatility (per-period sigma)
mlr_status mlr_ewma_vol(const double *returns, size_t n, double lambda, double *out);

// GARCH(1,1): fit by Gaussian MLE, filter in-sample, forecast ahead
typedef struct {
    double omega, alpha, beta;  // sigma2[t] = omega + alpha*r[t-1]^2 + beta*sigma2[t-1]
    double sigma2_next;         // one-step-ahead variance after the fit sample
    double loglik;              // maximized log-likelihood
    int converged;              // 0 = best-effort result
} mlr_garch;

mlr_status mlr_garch_fit(const double *returns, size_t n, mlr_garch *model_out);
mlr_status mlr_garch_filter(const mlr_garch *m, const double *returns, size_t n, double *sigma_out);
mlr_status mlr_garch_forecast(const mlr_garch *m, size_t horizon, double *sigma_out);

// Range-based per-bar estimators (bad bars produce NAN, not errors)
mlr_status mlr_parkinson_vol(const double *high, const double *low, size_t n, double *out);
mlr_status mlr_garman_klass_vol(const double *open, const double *high,
                                const double *low, const double *close,
                                size_t n, double *out);
```

### Rolling Statistics (`include/mlrisk/rolling.h`)

```c
// O(n) sliding-window mean and standard deviation
mlr_status mlr_rolling_mean(const double *x, size_t n, size_t window, double *out);
mlr_status mlr_rolling_std(const double *x, size_t n, size_t window, double *out);
```

### Position Sizing (`include/mlrisk/sizing.h`)

```c
// Volatility targeting: position = (target_vol / sigma) * (equity / price),
// capped at max_leverage * equity notional. To cap risk per position instead,
// pass the risk cap (fraction of equity) as target_vol - the formula is identical.
mlr_status mlr_vol_target_position(
    const double *sigma, double target_vol, double equity,
    const double *price, double max_leverage, size_t n, double *position_out);

// Kelly fraction: f = fraction * mean / sample_variance (may be negative)
mlr_status mlr_kelly_fraction(const double *returns, size_t n, double fraction, double *f_out);

// Drawdown-based exposure scaling: 1 at zero drawdown, linearly to 0 at max_dd
mlr_status mlr_drawdown_scale(const double *equity, size_t n, double max_dd, double *scale_out);
```

### Walk-Forward Splits (`include/mlrisk/split.h`)

```c
typedef struct {            // all ranges half-open [start, end)
    size_t train_start, train_end;         // pre-test training (purged)
    size_t test_start, test_end;           // test window
    size_t train_post_start, train_post_end; // post-test training (opt-in, embargoed)
} mlr_split;

mlr_status mlr_walk_forward_splits(
    size_t n, size_t train_len, size_t test_len, size_t step,
    size_t purge, size_t embargo, int include_post_train,
    mlr_split *splits_out, size_t capacity, size_t *count_out);
```

- `purge` drops the last samples of each training window whose labels would overlap the test horizon.
- `embargo` skips samples immediately after the test window before any post-test training data.
- `include_post_train` exposes a post-test training segment for purged-CV-style model selection. **Warning:** that segment is future data relative to the test window — leave it at 0 for genuine out-of-sample walk-forward.
- Pass `splits_out = NULL` for a count query; insufficient `capacity` returns `MLR_EBOUNDS` with the required count in `*count_out`.

### Linear Regression (`include/mlrisk/linreg.h`)

```c
mlr_lin_model model;
mlr_lin_model_init(&model, feature_dim, ridge_param);

mlr_status mlr_linreg_fit(const double *X, const double *y, size_t n, size_t d,
                          double ridge, mlr_lin_model *model_out);
mlr_status mlr_linreg_predict(const double *X, size_t n, size_t d,
                              const mlr_lin_model *model, double *out);

mlr_lin_model_free(&model);
```

### Error Handling

All functions return `mlr_status`:

- `MLR_OK` - Success
- `MLR_EINVAL` - Invalid argument
- `MLR_ENOMEM` - Memory allocation failure
- `MLR_EBOUNDS` - Output capacity too small (splits)
- `MLR_EDOMAIN` - Domain error (singular matrix, zero variance, bad equity path)

## Conventions

### Volatility

mlrisk uses **per-period volatility** throughout. To convert annualized volatility to per-period:

```
per_period_vol = annualized_vol / sqrt(periods_per_year)
```

- Daily data: `daily_vol = annualized_vol / sqrt(252)`
- Weekly data: `weekly_vol = annualized_vol / sqrt(52)`
- Monthly data: `monthly_vol = annualized_vol / sqrt(12)`

### Variance

Each function documents its variance convention explicitly:

- `mlr_rolling_std` uses **population** variance (divides by `window`) — note pandas `rolling().std()` defaults to the sample convention.
- `mlr_kelly_fraction` uses **sample** variance (divides by `n-1`).
- `mlr_garch_fit` is Gaussian **MLE**; returns are assumed mean-zero and the recursion is seeded with the mean of squared returns.

### EWMA Parameters

Typical `lambda` values for daily data: 0.94 - 0.97. Higher lambda = slower decay = more persistent volatility.

### Lookahead Avoidance

`mlr_walk_forward_splits` guarantees training data precedes test data with no overlap, and additionally supports purging (drop training labels that overlap the test horizon) and embargo. When backtesting, always use walk-forward splits — and purge when your labels span multiple periods.

## v2 Breaking Changes

v2.0.0 reworked the public API; there is no compatibility layer.

| v1 | v2 |
|---|---|
| `rolling_mean` / `rolling_std` / `ewma_vol` | `mlr_rolling_mean` / `mlr_rolling_std` / `mlr_ewma_vol` |
| `vol_target_position` | `mlr_vol_target_position` (now validates `target_vol > 0`) |
| `risk_cap_position` | removed — identical to `mlr_vol_target_position`; pass the risk cap as `target_vol` |
| `risk_forecast_ewma` | removed — was an alias for `mlr_ewma_vol` |
| `walk_forward_ranges` + `mlr_range` | `mlr_walk_forward_splits` + `mlr_split` (capacity-safe, purge/embargo) |
| `linreg_fit` / `linreg_predict` | `mlr_linreg_fit` / `mlr_linreg_predict` |
| `mlrisk/risk.h` | removed — `mlr_lin_model` now lives in `mlrisk/linreg.h` |

## License

MIT - see [LICENSE](LICENSE).
