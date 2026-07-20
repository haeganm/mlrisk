#ifndef MLRISK_SPLIT_H
#define MLRISK_SPLIT_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file split.h
 * @brief Purged + embargoed walk-forward splits for time-series cross-validation
 *
 * Generates train/test splits that avoid lookahead bias. Purging drops the
 * training samples closest to the test window (whose labels overlap the test
 * horizon); the embargo skips samples immediately after the test window before
 * any post-test training data.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief One walk-forward split. All ranges are half-open [start, end).
 */
typedef struct {
    size_t train_start;      /**< Pre-test training start */
    size_t train_end;        /**< Pre-test training end == test_start - purge */
    size_t test_start;       /**< Test window start */
    size_t test_end;         /**< Test window end */
    size_t train_post_start; /**< Post-test training start >= test_end + embargo;
                                  segment is empty (start == end) unless
                                  include_post_train was set */
    size_t train_post_end;   /**< Post-test training end (== n when non-empty) */
} mlr_split;

/**
 * @brief Generate purged + embargoed walk-forward splits
 *
 * Split i places a train_len-sample training window starting at i*step,
 * followed immediately by a test_len-sample test window. `purge` drops the
 * last `purge` training samples before the test window.
 *
 * If `include_post_train` is nonzero, each split also exposes a post-test
 * training segment [test_end + embargo, n) for purged-CV-style model
 * selection. WARNING: that segment is future data relative to the test
 * window - using it for anything the test evaluation depends on leaks.
 * Leave include_post_train at 0 for genuine out-of-sample walk-forward
 * (post segments are then always empty).
 *
 * Count-query mode: pass splits_out == NULL to receive the required split
 * count in *count_out without writing any splits. Typical usage:
 *
 *   size_t count;
 *   mlr_walk_forward_splits(n, 252, 21, 21, 5, 5, 0, NULL, 0, &count);
 *   mlr_split *splits = malloc(count * sizeof *splits);
 *   mlr_walk_forward_splits(n, 252, 21, 21, 5, 5, 0, splits, count, &count);
 *
 * @param n         Total number of samples
 * @param train_len Training window length (must be > purge)
 * @param test_len  Test window length (>= 1)
 * @param step      Step between consecutive splits (>= 1)
 * @param purge     Samples trimmed from the end of each training window
 * @param embargo   Samples skipped after each test window before post-test training
 * @param include_post_train Nonzero to populate post-test training segments
 * @param splits_out Output array, or NULL for a count query
 * @param capacity  Length of splits_out (ignored when splits_out == NULL)
 * @param count_out Receives the required/generated split count (must be non-NULL)
 * @return MLR_OK on success (count 0 is not an error), MLR_EINVAL on invalid
 *         parameters, MLR_EBOUNDS if capacity < required count (the first
 *         `capacity` splits are written and *count_out is the full count)
 */
mlr_status mlr_walk_forward_splits(
    size_t n,
    size_t train_len,
    size_t test_len,
    size_t step,
    size_t purge,
    size_t embargo,
    int include_post_train,
    mlr_split *splits_out,
    size_t capacity,
    size_t *count_out
);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_SPLIT_H */
