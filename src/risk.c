#include "mlrisk/risk.h"
#include "mlrisk/rolling.h"
#include <stdlib.h>
#include <string.h>

mlr_status mlr_lin_model_init(mlr_lin_model *model, size_t d, double ridge) {
    if (model == NULL || d == 0) {
        return MLR_EINVAL;
    }

    model->d = d;
    model->ridge = ridge;
    model->b = 0.0;
    model->w = (double *)calloc(d, sizeof(double));
    
    if (model->w == NULL) {
        return MLR_ENOMEM;
    }

    return MLR_OK;
}

void mlr_lin_model_free(mlr_lin_model *model) {
    if (model != NULL) {
        free(model->w);
        model->w = NULL;
        model->d = 0;
    }
}

mlr_status risk_forecast_ewma(const double *returns, size_t n, double lambda, double *sigma_out) {
    return ewma_vol(returns, n, lambda, sigma_out);
}
