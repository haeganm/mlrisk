#include "mlrisk/mlrisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 200
#define LOW_VOL_PERIOD 100
#define HIGH_VOL_MULTIPLIER 3.0

/**
 * Generate synthetic returns with volatility regime change
 */
static void generate_synthetic_returns(double *returns, size_t n) {
    srand((unsigned int)time(NULL));
    
    // Low volatility regime (first half)
    for (size_t i = 0; i < LOW_VOL_PERIOD; i++) {
        returns[i] = 0.001 * ((double)rand() / RAND_MAX - 0.5);
    }
    
    // High volatility regime (second half)
    for (size_t i = LOW_VOL_PERIOD; i < n; i++) {
        returns[i] = HIGH_VOL_MULTIPLIER * 0.001 * ((double)rand() / RAND_MAX - 0.5);
    }
}

/**
 * Generate synthetic prices (random walk)
 */
static void generate_synthetic_prices(const double *returns, double *prices, size_t n, double initial_price) {
    prices[0] = initial_price;
    for (size_t i = 1; i < n; i++) {
        prices[i] = prices[i-1] * (1.0 + returns[i]);
    }
}

/**
 * Compute summary statistics
 */
static void compute_stats(const double *data, size_t n, double *mean, double *min, double *max) {
    double sum = 0.0;
    *min = data[0];
    *max = data[0];
    
    size_t valid_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (!mlr_isnan(data[i])) {
            sum += data[i];
            valid_count++;
            if (data[i] < *min) *min = data[i];
            if (data[i] > *max) *max = data[i];
        }
    }
    
    *mean = (valid_count > 0) ? (sum / valid_count) : 0.0;
}

int main(void) {
    printf("=== mlrisk Volatility Targeting Demo ===\n\n");
    
    // Allocate arrays
    double returns[N];
    double prices[N];
    double sigma[N];
    double positions[N];
    
    // Generate synthetic data
    printf("Generating synthetic returns with volatility regime change...\n");
    printf("  Period 0-%zu: Low volatility regime\n", (size_t)(LOW_VOL_PERIOD - 1));
    printf("  Period %zu-%zu: High volatility regime (%.1fx multiplier)\n\n",
           (size_t)LOW_VOL_PERIOD, (size_t)(N - 1), HIGH_VOL_MULTIPLIER);
    
    generate_synthetic_returns(returns, N);
    generate_synthetic_prices(returns, prices, N, 100.0);
    
    // Compute EWMA volatility
    printf("Computing EWMA volatility (lambda=0.94)...\n");
    double lambda = 0.94;
    mlr_status status = mlr_ewma_vol(returns, N, lambda, sigma);
    
    if (status != MLR_OK) {
        fprintf(stderr, "Error computing EWMA volatility: %d\n", status);
        return 1;
    }
    
    // Compute position sizing
    printf("Computing position sizes (volatility targeting)...\n");
    double target_vol = 0.01;  // 1% per-period target volatility
    double equity = 100000.0;   // $100k account
    double max_leverage = 2.0;  // 2x maximum leverage
    
    status = mlr_vol_target_position(
        sigma, target_vol, equity, prices, max_leverage, N, positions
    );
    
    if (status != MLR_OK) {
        fprintf(stderr, "Error computing position sizes: %d\n", status);
        return 1;
    }
    
    // Print sample rows
    printf("\n=== Sample Data (first 10 and last 10 periods) ===\n");
    printf("%6s %10s %10s %10s %12s %12s\n",
           "Period", "Return", "Price", "Sigma", "Position", "Notional");
    printf("%6s %10s %10s %10s %12s %12s\n",
           "------", "------", "-----", "-----", "--------", "--------");
    
    // First 10 rows
    for (size_t i = 0; i < 10 && i < N; i++) {
        double notional = positions[i] * prices[i];
        printf("%6zu %10.6f %10.2f %10.6f %12.2f %12.2f\n",
               i, returns[i], prices[i], sigma[i], positions[i], notional);
    }
    
    printf("  ...\n");
    
    // Last 10 rows
    size_t start = (N > 10) ? (N - 10) : 0;
    for (size_t i = start; i < N; i++) {
        double notional = positions[i] * prices[i];
        printf("%6zu %10.6f %10.2f %10.6f %12.2f %12.2f\n",
               i, returns[i], prices[i], sigma[i], positions[i], notional);
    }
    
    // Summary statistics
    printf("\n=== Summary Statistics ===\n");
    
    // Sigma stats
    double sigma_mean, sigma_min, sigma_max;
    compute_stats(sigma, N, &sigma_mean, &sigma_min, &sigma_max);
    printf("Volatility (sigma):\n");
    printf("  Mean: %.6f\n", sigma_mean);
    printf("  Min:  %.6f\n", sigma_min);
    printf("  Max:  %.6f\n", sigma_max);
    
    // Position stats
    double pos_mean, pos_min, pos_max;
    compute_stats(positions, N, &pos_mean, &pos_min, &pos_max);
    printf("\nPosition sizes:\n");
    printf("  Mean: %.2f shares\n", pos_mean);
    printf("  Min:  %.2f shares\n", pos_min);
    printf("  Max:  %.2f shares\n", pos_max);
    
    // Compare low vol vs high vol periods
    printf("\n=== Regime Comparison ===\n");
    
    double low_vol_sigma_mean = 0.0, high_vol_sigma_mean = 0.0;
    double low_vol_pos_mean = 0.0, high_vol_pos_mean = 0.0;
    size_t low_count = 0, high_count = 0;
    
    for (size_t i = 0; i < LOW_VOL_PERIOD; i++) {
        if (!mlr_isnan(sigma[i])) {
            low_vol_sigma_mean += sigma[i];
            low_vol_pos_mean += positions[i];
            low_count++;
        }
    }
    
    for (size_t i = LOW_VOL_PERIOD; i < N; i++) {
        if (!mlr_isnan(sigma[i])) {
            high_vol_sigma_mean += sigma[i];
            high_vol_pos_mean += positions[i];
            high_count++;
        }
    }
    
    if (low_count > 0) {
        low_vol_sigma_mean /= low_count;
        low_vol_pos_mean /= low_count;
    }
    if (high_count > 0) {
        high_vol_sigma_mean /= high_count;
        high_vol_pos_mean /= high_count;
    }
    
    printf("Low volatility period (0-%zu):\n", (size_t)(LOW_VOL_PERIOD - 1));
    printf("  Avg sigma:  %.6f\n", low_vol_sigma_mean);
    printf("  Avg position: %.2f shares\n", low_vol_pos_mean);
    
    printf("\nHigh volatility period (%zu-%zu):\n", (size_t)LOW_VOL_PERIOD, (size_t)(N - 1));
    printf("  Avg sigma:  %.6f\n", high_vol_sigma_mean);
    printf("  Avg position: %.2f shares\n", high_vol_pos_mean);
    
    printf("\nRatio (high/low):\n");
    if (low_vol_sigma_mean > 0) {
        printf("  Sigma:  %.2fx\n", high_vol_sigma_mean / low_vol_sigma_mean);
    }
    if (low_vol_pos_mean > 0) {
        printf("  Position: %.2fx (inverse relationship expected)\n",
               high_vol_pos_mean / low_vol_pos_mean);
    }
    
    printf("\n=== Demo Complete ===\n");
    printf("Key observations:\n");
    printf("  - Sigma increases in high volatility regime\n");
    printf("  - Position sizes decrease as sigma increases (volatility targeting)\n");
    printf("  - Leverage cap (%.1fx) prevents excessive exposure\n", max_leverage);
    
    return 0;
}
