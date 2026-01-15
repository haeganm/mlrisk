#include "mlrisk/split.h"

size_t walk_forward_ranges(
    size_t n,
    size_t train_len,
    size_t test_len,
    size_t step,
    mlr_range *ranges_out
) {
    if (ranges_out == NULL) {
        return 0;
    }
    if (train_len == 0 || test_len == 0 || step == 0) {
        return 0;
    }
    if (train_len + test_len > n) {
        return 0;
    }

    size_t count = 0;
    size_t train_start = 0;

    while (train_start + train_len + test_len <= n) {
        mlr_range range;
        range.train_start = train_start;
        range.train_end = train_start + train_len;
        range.test_start = train_start + train_len;
        range.test_end = train_start + train_len + test_len;

        ranges_out[count] = range;
        count++;

        train_start += step;

        // Check if we can fit another range
        if (train_start + train_len + test_len > n) {
            break;
        }
    }

    return count;
}
