#ifndef MLRISK_RISK_H
#define MLRISK_RISK_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file risk.h
 * @brief Volatility forecasting API
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Linear model structure for volatility forecasting
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
 * @brief Forecast volatility using EWMA
 * 
 * Wrapper around ewma_vol() for consistency with risk forecasting API.
 * 
 * @param returns Array of returns (length n)
 * @param n Length of returns array
 * @param lambda EWMA decay factor (typically 0.94-0.97)
 * @param sigma_out Output array of per-period sigma values (length n, must be pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status risk_forecast_ewma(const double *returns, size_t n, double lambda, double *sigma_out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_RISK_H */
