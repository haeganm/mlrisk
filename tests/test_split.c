#include "mlrisk/split.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

static int test_split_purged_embargoed_known_answer(void) {
    mlr_split splits[8];
    size_t count = 0;

    mlr_status status = mlr_walk_forward_splits(
        60, 20, 10, 10, 3, 5, 1, splits, 8, &count
    );
    ASSERT(status == MLR_OK, "purged split should return MLR_OK");
    ASSERT(count == 4, "n=60,train=20,test=10,step=10 should yield 4 splits");

    // Split 0: train [0,17), test [20,30), post [35,60)
    ASSERT(splits[0].train_start == 0 && splits[0].train_end == 17, "split0 train [0,17)");
    ASSERT(splits[0].test_start == 20 && splits[0].test_end == 30, "split0 test [20,30)");
    ASSERT(splits[0].train_post_start == 35 && splits[0].train_post_end == 60, "split0 post [35,60)");

    // Split 2: train [20,37), test [40,50), post [55,60)
    ASSERT(splits[2].train_start == 20 && splits[2].train_end == 37, "split2 train [20,37)");
    ASSERT(splits[2].test_start == 40 && splits[2].test_end == 50, "split2 test [40,50)");
    ASSERT(splits[2].train_post_start == 55 && splits[2].train_post_end == 60, "split2 post [55,60)");

    // Split 3: test [50,60); embargo pushes post past n -> empty
    ASSERT(splits[3].test_end == 60, "split3 test_end == n");
    ASSERT(splits[3].train_post_start == splits[3].train_post_end, "split3 post should be empty");

    printf("  PASS: purged+embargoed known answer\n");
    return 0;
}

static int test_split_v1_degeneration(void) {
    // purge=0, no post segments: must reproduce the v1 walk-forward behavior
    mlr_split splits[20];
    size_t count = 0;

    mlr_status status = mlr_walk_forward_splits(
        100, 20, 10, 5, 0, 0, 0, splits, 20, &count
    );
    ASSERT(status == MLR_OK, "v1-style split should return MLR_OK");
    ASSERT(count > 0, "should generate at least one split");

    ASSERT(splits[0].train_start == 0, "first split train_start should be 0");
    ASSERT(splits[0].train_end == 20, "first split train_end should be 20");
    ASSERT(splits[0].test_start == 20, "first split test_start should be 20");
    ASSERT(splits[0].test_end == 30, "first split test_end should be 30");

    ASSERT(splits[1].train_start == 5, "second split train_start should be 5");
    ASSERT(splits[1].train_end == 25, "second split train_end should be 25");
    ASSERT(splits[1].test_start == 25, "second split test_start should be 25");
    ASSERT(splits[1].test_end == 35, "second split test_end should be 35");

    // Boundary case: last split ends exactly at n
    status = mlr_walk_forward_splits(50, 20, 10, 5, 0, 0, 0, splits, 20, &count);
    ASSERT(status == MLR_OK, "boundary split should return MLR_OK");
    ASSERT(count == 5, "n=50,train=20,test=10,step=5 should yield 5 splits");
    ASSERT(splits[count - 1].test_end == 50, "last split test_end should equal n");

    printf("  PASS: v1 degeneration (purge=0)\n");
    return 0;
}

static int test_split_invariants(void) {
    mlr_split splits[32];
    size_t count = 0;

    mlr_status status = mlr_walk_forward_splits(
        200, 40, 15, 7, 5, 3, 1, splits, 32, &count
    );
    ASSERT(status == MLR_OK, "invariant run should return MLR_OK");
    ASSERT(count > 0 && count <= 32, "invariant run should fit in capacity");

    for (size_t i = 0; i < count; i++) {
        ASSERT(splits[i].train_end == splits[i].test_start - 5,
               "train_end should equal test_start - purge");
        ASSERT(splits[i].train_start < splits[i].train_end, "train segment should be non-empty");
        ASSERT(splits[i].test_start < splits[i].test_end, "test segment should be non-empty");
        ASSERT(splits[i].test_end <= 200, "test_end should be within bounds");
        ASSERT(splits[i].train_post_start <= splits[i].train_post_end, "post segment well-formed");
        if (splits[i].train_post_start < splits[i].train_post_end) {
            ASSERT(splits[i].train_post_start >= splits[i].test_end + 3,
                   "post segment should respect the embargo");
            ASSERT(splits[i].train_post_end <= 200, "post segment within bounds");
        }
    }

    printf("  PASS: split invariants\n");
    return 0;
}

static int test_split_count_query_and_capacity(void) {
    size_t full_count = 0;
    mlr_status status = mlr_walk_forward_splits(
        100, 20, 10, 5, 2, 1, 0, NULL, 0, &full_count
    );
    ASSERT(status == MLR_OK, "count query should return MLR_OK");
    ASSERT(full_count > 1, "count query should find multiple splits");

    // Full-capacity call agrees with the count query
    mlr_split *splits = (mlr_split *)malloc(full_count * sizeof(mlr_split));
    ASSERT(splits != NULL, "allocation should succeed");
    size_t count = 0;
    status = mlr_walk_forward_splits(100, 20, 10, 5, 2, 1, 0, splits, full_count, &count);
    ASSERT(status == MLR_OK, "full-capacity call should return MLR_OK");
    ASSERT(count == full_count, "full-capacity count should match count query");
    free(splits);

    // Insufficient capacity: exactly `capacity` entries written, EBOUNDS returned
    size_t cap = full_count - 1;
    mlr_split *small = (mlr_split *)malloc((cap + 1) * sizeof(mlr_split));
    ASSERT(small != NULL, "allocation should succeed");
    memset(small, 0xAB, (cap + 1) * sizeof(mlr_split));
    mlr_split sentinel = small[cap];

    count = 0;
    status = mlr_walk_forward_splits(100, 20, 10, 5, 2, 1, 0, small, cap, &count);
    ASSERT(status == MLR_EBOUNDS, "insufficient capacity should return MLR_EBOUNDS");
    ASSERT(count == full_count, "count_out should report the full required count");
    ASSERT(memcmp(&small[cap], &sentinel, sizeof(mlr_split)) == 0,
           "entry past capacity must be untouched");
    ASSERT(small[0].train_start == 0 && small[0].test_start == 20,
           "entries within capacity should be written");
    free(small);

    printf("  PASS: count query and capacity handling\n");
    return 0;
}

static int test_split_invalid_inputs(void) {
    mlr_split splits[4];
    size_t count = 123;

    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 0, 0, 0, splits, 4, NULL) == MLR_EINVAL,
           "NULL count_out should return MLR_EINVAL");
    ASSERT(mlr_walk_forward_splits(100, 0, 10, 5, 0, 0, 0, splits, 4, &count) == MLR_EINVAL,
           "train_len=0 should return MLR_EINVAL");
    ASSERT(mlr_walk_forward_splits(100, 20, 0, 5, 0, 0, 0, splits, 4, &count) == MLR_EINVAL,
           "test_len=0 should return MLR_EINVAL");
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 0, 0, 0, 0, splits, 4, &count) == MLR_EINVAL,
           "step=0 should return MLR_EINVAL");
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 20, 0, 0, splits, 4, &count) == MLR_EINVAL,
           "purge >= train_len should return MLR_EINVAL");

    // Not enough data is not an error: zero splits
    count = 123;
    ASSERT(mlr_walk_forward_splits(10, 20, 10, 5, 0, 0, 0, splits, 4, &count) == MLR_OK,
           "train+test > n should return MLR_OK");
    ASSERT(count == 0, "train+test > n should yield zero splits");

    printf("  PASS: split invalid inputs\n");
    return 0;
}

int test_split(void) {
    int failures = 0;
    failures += test_split_purged_embargoed_known_answer();
    failures += test_split_v1_degeneration();
    failures += test_split_invariants();
    failures += test_split_count_query_and_capacity();
    failures += test_split_invalid_inputs();
    return failures;
}
