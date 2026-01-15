#include "mlrisk/linreg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper: Solve linear system Ax = b using Gaussian elimination with partial pivoting
// A is d x d, b is length d, x is output (length d)
static mlr_status solve_linear_system(double *A, double *b, size_t d, double *x) {
    // Create augmented matrix [A|b]
    double *aug = (double *)malloc(d * (d + 1) * sizeof(double));
    if (aug == NULL) {
        return MLR_ENOMEM;
    }

    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            aug[i * (d + 1) + j] = A[i * d + j];
        }
        aug[i * (d + 1) + d] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for (size_t col = 0; col < d; col++) {
        // Find pivot
        size_t max_row = col;
        double max_val = fabs(aug[col * (d + 1) + col]);
        for (size_t row = col + 1; row < d; row++) {
            double val = fabs(aug[row * (d + 1) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Check for singular matrix
        if (max_val < 1e-10) {
            free(aug);
            return MLR_EDOMAIN;
        }

        // Swap rows
        if (max_row != col) {
            for (size_t j = 0; j < d + 1; j++) {
                double temp = aug[col * (d + 1) + j];
                aug[col * (d + 1) + j] = aug[max_row * (d + 1) + j];
                aug[max_row * (d + 1) + j] = temp;
            }
        }

        // Eliminate
        for (size_t row = col + 1; row < d; row++) {
            double factor = aug[row * (d + 1) + col] / aug[col * (d + 1) + col];
            for (size_t j = col; j < d + 1; j++) {
                aug[row * (d + 1) + j] -= factor * aug[col * (d + 1) + j];
            }
        }
    }

    // Back substitution
    for (int i = (int)d - 1; i >= 0; i--) {
        x[i] = aug[i * (d + 1) + d];
        for (size_t j = i + 1; j < d; j++) {
            x[i] -= aug[i * (d + 1) + j] * x[j];
        }
        x[i] /= aug[i * (d + 1) + i];
    }

    free(aug);
    return MLR_OK;
}

mlr_status linreg_fit(
    const double *X,
    const double *y,
    size_t n,
    size_t d,
    double ridge,
    mlr_lin_model *model_out
) {
    if (X == NULL || y == NULL || model_out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0 || d == 0) {
        return MLR_EINVAL;
    }
    if (model_out->d != d) {
        return MLR_EINVAL;
    }

    // Compute means: mu_x[j] for each feature, mu_y for target
    double *mu_x = (double *)calloc(d, sizeof(double));
    if (mu_x == NULL) {
        return MLR_ENOMEM;
    }

    double mu_y = 0.0;
    for (size_t i = 0; i < n; i++) {
        mu_y += y[i];
        for (size_t j = 0; j < d; j++) {
            mu_x[j] += X[i * d + j];
        }
    }
    mu_y /= n;
    for (size_t j = 0; j < d; j++) {
        mu_x[j] /= n;
    }

    // Build normal equations on centered data: A = Xc^T Xc + alpha * I
    double *XtX = (double *)calloc(d * d, sizeof(double));
    if (XtX == NULL) {
        free(mu_x);
        return MLR_ENOMEM;
    }

    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                double xc_i = X[k * d + i] - mu_x[i];
                double xc_j = X[k * d + j] - mu_x[j];
                sum += xc_i * xc_j;
            }
            XtX[i * d + j] = sum;
        }
        // Add ridge regularization (not applied to intercept)
        XtX[i * d + i] += ridge;
    }

    // Compute rhs = Xc^T yc (centered)
    double *Xty = (double *)calloc(d, sizeof(double));
    if (Xty == NULL) {
        free(XtX);
        free(mu_x);
        return MLR_ENOMEM;
    }

    for (size_t i = 0; i < d; i++) {
        double sum = 0.0;
        for (size_t k = 0; k < n; k++) {
            double xc_i = X[k * d + i] - mu_x[i];
            double yc = y[k] - mu_y;
            sum += xc_i * yc;
        }
        Xty[i] = sum;
    }

    // Solve A w = rhs
    mlr_status status = solve_linear_system(XtX, Xty, d, model_out->w);
    
    // Set intercept: b = mu_y - sum_j(mu_x[j] * w[j])
    if (status == MLR_OK) {
        double xw = 0.0;
        for (size_t j = 0; j < d; j++) {
            xw += mu_x[j] * model_out->w[j];
        }
        model_out->b = mu_y - xw;
    }

    free(XtX);
    free(Xty);
    free(mu_x);

    return status;
}

mlr_status linreg_predict(
    const double *X,
    size_t n,
    size_t d,
    const mlr_lin_model *model,
    double *out
) {
    if (X == NULL || model == NULL || out == NULL) {
        return MLR_EINVAL;
    }
    if (n == 0 || d == 0) {
        return MLR_EINVAL;
    }
    if (model->d != d) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < n; i++) {
        double pred = model->b;
        for (size_t j = 0; j < d; j++) {
            pred += X[i * d + j] * model->w[j];
        }
        out[i] = pred;
    }

    return MLR_OK;
}
