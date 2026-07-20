#include "mlrisk/split.h"

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
) {
    if (count_out == NULL) {
        return MLR_EINVAL;
    }
    if (train_len == 0 || test_len == 0 || step == 0) {
        return MLR_EINVAL;
    }
    if (purge >= train_len) {
        return MLR_EINVAL;
    }

    size_t count = 0;
    for (size_t train_start = 0;
         train_start + train_len + test_len <= n;
         train_start += step) {
        if (splits_out != NULL && count < capacity) {
            mlr_split s;
            s.test_start = train_start + train_len;
            s.test_end = s.test_start + test_len;
            s.train_start = train_start;
            s.train_end = s.test_start - purge;

            if (include_post_train && s.test_end + embargo < n) {
                s.train_post_start = s.test_end + embargo;
                s.train_post_end = n;
            } else {
                s.train_post_start = n;
                s.train_post_end = n;
            }

            splits_out[count] = s;
        }
        count++;
    }

    *count_out = count;
    if (splits_out != NULL && count > capacity) {
        return MLR_EBOUNDS;
    }
    return MLR_OK;
}
