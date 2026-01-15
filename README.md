# mlrisk

> A lightweight C11 library for volatility forecasting and position sizing with time-series-safe evaluation utilities.

[![C11](https://img.shields.io/badge/C-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![CMake](https://img.shields.io/badge/CMake-3.15+-green.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Documentation](#documentation)
- [License](#license)

## Overview

**mlrisk** is a production-ready C11 static library designed for quantitative finance applications. It provides essential tools for:

- 📊 **Volatility Forecasting**: EWMA-based volatility estimation with optional linear regression models
- 📈 **Position Sizing**: Volatility targeting and risk capping with leverage constraints
- 🔄 **Time-Series Utilities**: Walk-forward splits for backtesting without lookahead bias
- 📉 **Rolling Statistics**: Numerically stable rolling mean, standard deviation, and EWMA calculations

Perfect for algorithmic trading systems, risk management tools, and quantitative research.

## Features

✨ **Production Ready**
- Pure C11, zero external dependencies
- Cross-platform (Windows, macOS, Linux)
- Comprehensive unit tests with 100% pass rate
- Numerically stable algorithms (Welford's method)

🛡️ **Robust & Safe**
- Complete input validation and error handling
- Time-series safe (no lookahead bias)
- Memory-safe with proper resource management

⚡ **Performance**
- Static library for minimal overhead
- Optimized for small to medium datasets
- Efficient walk-forward evaluation utilities

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mlrisk.git
cd mlrisk
```

### 2. Build the Library

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build .
```

### 3. Run Tests

```bash
# Run all tests
ctest

# Or run directly
./mlrisk_tests  # Linux/macOS
.\Debug\mlrisk_tests.exe  # Windows
```

### 4. Try the Demo

```bash
./vol_target_demo  # Linux/macOS
.\Debug\vol_target_demo.exe  # Windows
```

## Installation

### Prerequisites

- **CMake** 3.15 or higher ([Download](https://cmake.org/download/))
- **C11-compatible compiler**:
  - GCC 4.9+ or Clang 3.3+ (Linux/macOS)
  - MSVC 2015+ (Windows)
  - Or any compiler supporting C11 standard

### Build Instructions

#### Linux / macOS

```bash
mkdir build
cd build
cmake ..
make
```

#### Windows (Visual Studio)

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

#### Windows (MinGW)

```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
cmake --build .
```

### Running Tests

```bash
# From build directory
ctest

# Or with verbose output
ctest --output-on-failure

# For Visual Studio multi-config generators
ctest -C Debug
ctest -C Release
```

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
    double sigma[n];
    mlr_status status = risk_forecast_ewma(returns, n, 0.94, sigma);
    if (status != MLR_OK) {
        fprintf(stderr, "Error computing volatility\n");
        return 1;
    }
    
    // 2. Compute position sizes using volatility targeting
    double positions[n];
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
# Find or add mlrisk
add_subdirectory(path/to/mlrisk)

# Link to your target
target_link_libraries(your_target PRIVATE mlrisk)
target_include_directories(your_target PRIVATE path/to/mlrisk/include)
```

## API Reference

### Core Modules

#### Rolling Statistics (`include/mlrisk/rolling.h`)

```c
// Compute rolling mean
mlr_status rolling_mean(const double *x, size_t n, size_t window, double *out);

// Compute rolling standard deviation (numerically stable)
mlr_status rolling_std(const double *x, size_t n, size_t window, double *out);

// Compute EWMA volatility (per-period sigma)
mlr_status ewma_vol(const double *returns, size_t n, double lambda, double *out);
```

#### Risk Forecasting (`include/mlrisk/risk.h`)

```c
// EWMA-based volatility forecast
mlr_status risk_forecast_ewma(const double *returns, size_t n, double lambda, double *sigma_out);

// Linear model for volatility forecasting
mlr_lin_model model;
mlr_lin_model_init(&model, feature_dim, ridge_param);
// ... use model ...
mlr_lin_model_free(&model);
```

#### Position Sizing (`include/mlrisk/sizing.h`)

```c
// Volatility targeting
mlr_status vol_target_position(
    const double *sigma,      // per-period volatility
    double target_vol,        // target per-period volatility
    double equity,            // account equity
    const double *price,      // asset prices
    double max_leverage,       // maximum leverage multiplier
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

#### Walk-Forward Splits (`include/mlrisk/split.h`)

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

#### Linear Regression (`include/mlrisk/linreg.h`)

```c
// Fit ridge regression model
mlr_status linreg_fit(
    const double *X,      // feature matrix (n rows, d columns)
    const double *y,      // target vector
    size_t n,             // number of samples
    size_t d,              // number of features
    double ridge,          // ridge regularization parameter
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

## Documentation

### Volatility Conventions

**Important**: mlrisk uses **per-period volatility** throughout.

To convert annualized volatility to per-period:

```
per_period_vol = annualized_vol / sqrt(periods_per_year)
```

**Examples:**
- Daily data: `daily_vol = annualized_vol / sqrt(252)`
- Weekly data: `weekly_vol = annualized_vol / sqrt(52)`
- Monthly data: `monthly_vol = annualized_vol / sqrt(12)`

### EWMA Parameters

Typical `lambda` values:
- **Daily data**: 0.94 - 0.97
- **Higher lambda** = slower decay = more persistent volatility

### Lookahead Avoidance

The `walk_forward_ranges()` function ensures no lookahead bias:
- ✅ Training data always comes **before** test data
- ✅ No overlap between training and test sets
- ✅ `test_start >= train_end` for all ranges

**Best Practice**: When backtesting, always use walk-forward splits to avoid overfitting and unrealistic performance estimates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Made with ❤️ for quantitative finance**
