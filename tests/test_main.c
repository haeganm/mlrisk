#include <stdio.h>
#include <stdlib.h>

extern int test_rolling(void);
extern int test_risk(void);
extern int test_sizing(void);
extern int test_split(void);
extern int test_linreg(void);

int main(void) {
    int failures = 0;

    printf("Running mlrisk tests...\n\n");

    printf("=== Rolling Statistics Tests ===\n");
    failures += test_rolling();
    printf("\n");

    printf("=== Risk Forecasting Tests ===\n");
    failures += test_risk();
    printf("\n");

    printf("=== Position Sizing Tests ===\n");
    failures += test_sizing();
    printf("\n");

    printf("=== Walk-Forward Split Tests ===\n");
    failures += test_split();
    printf("\n");

    printf("=== Linear Regression Tests ===\n");
    failures += test_linreg();
    printf("\n");

    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("FAILED: %d test(s) failed\n", failures);
        return 1;
    }
}
