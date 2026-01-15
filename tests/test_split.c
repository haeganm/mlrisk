#include "mlrisk/split.h"
#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while (0)

static int test_walk_forward_basic(void) {
    size_t n = 100;
    size_t train_len = 20;
    size_t test_len = 10;
    size_t step = 5;
    
    // Calculate maximum possible ranges
    size_t max_ranges = 0;
    if (train_len + test_len <= n && step > 0) {
        max_ranges = ((n - (train_len + test_len)) / step) + 1;
    }
    
    mlr_range *ranges = (mlr_range *)calloc(max_ranges, sizeof(mlr_range));
    if (ranges == NULL) {
        printf("  FAIL: Memory allocation failed\n");
        return 1;
    }
    
    size_t count = walk_forward_ranges(n, train_len, test_len, step, ranges);
    
    ASSERT(count > 0, "walk_forward_ranges should return at least one range");
    
    // Check first range
    ASSERT(ranges[0].train_start == 0, "First range train_start should be 0");
    ASSERT(ranges[0].train_end == 20, "First range train_end should be 20");
    ASSERT(ranges[0].test_start == 20, "First range test_start should be 20");
    ASSERT(ranges[0].test_end == 30, "First range test_end should be 30");
    
    // Check second range
    ASSERT(ranges[1].train_start == 5, "Second range train_start should be 5");
    ASSERT(ranges[1].train_end == 25, "Second range train_end should be 25");
    ASSERT(ranges[1].test_start == 25, "Second range test_start should be 25");
    ASSERT(ranges[1].test_end == 35, "Second range test_end should be 35");
    
    // Verify no lookahead: test_start >= train_end
    for (size_t i = 0; i < count; i++) {
        ASSERT(ranges[i].test_start >= ranges[i].train_end,
               "No lookahead: test_start should be >= train_end");
    }
    
    free(ranges);
    printf("  PASS: walk_forward_ranges basic\n");
    return 0;
}

static int test_walk_forward_boundaries(void) {
    mlr_range ranges[10];
    size_t n = 50;
    size_t train_len = 20;
    size_t test_len = 10;
    size_t step = 5;
    
    size_t count = walk_forward_ranges(n, train_len, test_len, step, ranges);
    
    // Should fit: [0,20) train, [20,30) test = 30 total
    // Next: [5,25) train, [25,35) test = 35 total
    // Next: [10,30) train, [30,40) test = 40 total
    // Next: [15,35) train, [35,45) test = 45 total
    // Next: [20,40) train, [40,50) test = 50 total (fits exactly)
    // Next: [25,45) train, [45,55) test = 55 total (doesn't fit)
    ASSERT(count == 5, "Should generate 5 ranges");
    
    // Last range should fit exactly
    ASSERT(ranges[count-1].test_end == n, "Last range test_end should equal n");
    
    printf("  PASS: walk_forward_ranges boundaries\n");
    return 0;
}

static int test_walk_forward_edge_cases(void) {
    mlr_range ranges[10];
    
    // NULL output
    size_t count = walk_forward_ranges(100, 20, 10, 5, NULL);
    ASSERT(count == 0, "walk_forward_ranges with NULL should return 0");
    
    // Zero train_len
    count = walk_forward_ranges(100, 0, 10, 5, ranges);
    ASSERT(count == 0, "walk_forward_ranges with train_len=0 should return 0");
    
    // Zero test_len
    count = walk_forward_ranges(100, 20, 0, 5, ranges);
    ASSERT(count == 0, "walk_forward_ranges with test_len=0 should return 0");
    
    // Zero step
    count = walk_forward_ranges(100, 20, 10, 0, ranges);
    ASSERT(count == 0, "walk_forward_ranges with step=0 should return 0");
    
    // train_len + test_len > n
    count = walk_forward_ranges(10, 20, 10, 5, ranges);
    ASSERT(count == 0, "walk_forward_ranges with train+test > n should return 0");
    
    printf("  PASS: walk_forward_ranges edge cases\n");
    return 0;
}

static int test_walk_forward_count(void) {
    mlr_range ranges[100];
    
    // Test various configurations
    size_t count1 = walk_forward_ranges(100, 20, 10, 5, ranges);
    ASSERT(count1 > 0, "Should generate ranges for n=100, train=20, test=10, step=5");
    
    size_t count2 = walk_forward_ranges(100, 50, 10, 10, ranges);
    ASSERT(count2 > 0, "Should generate ranges for n=100, train=50, test=10, step=10");
    ASSERT(count2 < count1, "Larger train window should generate fewer ranges");
    
    printf("  PASS: walk_forward_ranges count validation\n");
    return 0;
}

int test_split(void) {
    int failures = 0;
    failures += test_walk_forward_basic();
    failures += test_walk_forward_boundaries();
    failures += test_walk_forward_edge_cases();
    failures += test_walk_forward_count();
    return failures;
}
