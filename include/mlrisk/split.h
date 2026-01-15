#ifndef MLRISK_SPLIT_H
#define MLRISK_SPLIT_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file split.h
 * @brief Time-series walk-forward split utilities
 * 
 * Provides utilities for generating train/test splits that avoid lookahead bias.
 * All ranges are generated such that test data always comes after training data.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure representing a train/test range
 */
typedef struct {
    size_t train_start;  /**< Start index of training set (inclusive) */
    size_t train_end;    /**< End index of training set (exclusive) */
    size_t test_start;   /**< Start index of test set (inclusive) */
    size_t test_end;     /**< End index of test set (exclusive) */
} mlr_range;

/**
 * @brief Generate walk-forward ranges for time-series cross-validation
 * 
 * Generates a sequence of train/test splits that walk forward through the data.
 * Each split has:
 *   - Training set: [train_start, train_end)
 *   - Test set: [test_start, test_end)
 * 
 * The function ensures no lookahead: test_start >= train_end for all ranges.
 * 
 * Example with n=100, train_len=20, test_len=10, step=5:
 *   Range 0: train=[0,20), test=[20,30)
 *   Range 1: train=[5,25), test=[25,35)
 *   Range 2: train=[10,30), test=[30,40)
 *   ...
 * 
 * @param n Total length of time series
 * @param train_len Length of training window
 * @param test_len Length of test window
 * @param step Step size between ranges
 * @param ranges_out Output array of mlr_range structures (must be pre-allocated with sufficient size)
 * @return Number of valid ranges generated, or 0 on error
 */
size_t walk_forward_ranges(
    size_t n,
    size_t train_len,
    size_t test_len,
    size_t step,
    mlr_range *ranges_out
);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_SPLIT_H */
