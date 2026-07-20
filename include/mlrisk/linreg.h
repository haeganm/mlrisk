#ifndef MLRISK_LINREG_H
#define MLRISK_LINREG_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file linreg.h
 * @brief Ridge linear regression for volatility forecasting
 *
 * Provides closed-form ridge regression for small feature dimensions.
 * Uses Gaussian elimination with partial pivoting to solve normal equations.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Linear model structure
 */
typedef struct {
    size_t d;        /**< Feature dimension */
    double *w;       /**< Weight vector (length d) */
    double b;        /**< Bias/intercept */
    double ridge;    /**< Ridge regularization parameter */
} mlr_lin_model;

/**
 * @brief Initialize a linear model structure
 *
 * Allocates memory for weight vector w.
 *
 * @param model Pointer to model structure to initialize
 * @param d Feature dimension
 * @param ridge Ridge regularization parameter
 * @return MLR_OK on success, MLR_EINVAL on invalid input, MLR_ENOMEM on allocation failure
 */
mlr_status mlr_lin_model_init(mlr_lin_model *model, size_t d, double ridge);

/**
 * @brief Free memory allocated for a linear model
 *
 * @param model Pointer to model structure to free
 */
void mlr_lin_model_free(mlr_lin_model *model);

/**
 * @brief Fit a linear regression model using ridge regression
 * 
 * Solves the normal equations: (X^T X + alpha * I) w = X^T (y - b)
 * Uses Gaussian elimination with partial pivoting.
 * 
 * @param X Feature matrix (n rows, d columns), stored row-major: X[i*d + j] = X[i][j]
 * @param y Target vector (length n)
 * @param n Number of samples
 * @param d Number of features
 * @param ridge Ridge regularization parameter (alpha)
 * @param model_out Output model structure (must be initialized with mlr_lin_model_init)
 * @return MLR_OK on success, MLR_EINVAL on invalid input, MLR_EDOMAIN on singular matrix
 */
mlr_status mlr_linreg_fit(
    const double *X,
    const double *y,
    size_t n,
    size_t d,
    double ridge,
    mlr_lin_model *model_out
);

/**
 * @brief Predict using a linear regression model
 * 
 * Computes: y_pred = X * w + b
 * 
 * @param X Feature matrix (n rows, d columns), stored row-major
 * @param n Number of samples
 * @param d Number of features (must match model->d)
 * @param model Trained model
 * @param out Output predictions (length n, must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_linreg_predict(
    const double *X,
    size_t n,
    size_t d,
    const mlr_lin_model *model,
    double *out
);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_LINREG_H */
