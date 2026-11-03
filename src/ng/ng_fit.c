#include "ng_fit.h"
#include "residual_bridge.h"
#include <stdlib.h>
#include <string.h>

void vc_ng_fit_free(vc_ng_fit *fit)
{
    free(fit->params);
    free(fit->symbols);
    free(fit->base);
    memset(fit, 0, sizeof(*fit));
}

int vc_ng_fit_input(vc_ng_fit *fit, const uint8_t *src, const int *cfg,
                    const uint8_t *fixed, const uint8_t *mask, void *session,
                    uint16_t *activity, uint32_t *histogram)
{
    const size_t n = NG_N;
    uint8_t *feature = NULL, *classes = NULL;
    int ok = 0, rc;
    memset(fit, 0, sizeof(*fit));
    fit->params = malloc(65536);
    fit->symbols = malloc(n);
    feature = fixed ? malloc(n) : NULL;
    if (!activity) {
        classes = malloc(n);
        fit->base = malloc(n * sizeof(*fit->base));
    }
    if (!fit->params || !fit->symbols || (fixed && !feature) ||
        (!activity && (!classes || !fit->base))) goto done;
    for (size_t i = 0; fixed && i < n; ++i) {
        uint8_t delta = (uint8_t)((fixed[i] >> 1) ^ (0u - (fixed[i] & 1u)));
        feature[i] = (uint8_t)(src[i] - delta);
    }
    if (!fixed) {
        if (session || activity) goto done;
        rc = ng_plain_prepare_masked(src,cfg,mask,fit->params,65536,&fit->param_size,
                                     fit->symbols,classes,fit->base);
    } else if (session) {
        if (mask || activity) goto done;
        rc = ng_feature_prepare_cached(session, src, cfg, feature, fit->weights,
            fit->params, 65536, &fit->param_size, fit->symbols, classes, fit->base);
    } else if (activity) {
        if (mask || !histogram) goto done;
        rc = ng_feature_prepare_activity(src, cfg, feature, fit->weights,
            fit->params, 65536, &fit->param_size, fit->symbols, activity, histogram);
    } else {
        rc = ng_feature_prepare_masked(src, cfg, feature, mask, fit->weights,
            fit->params, 65536, &fit->param_size, fit->symbols, classes, fit->base);
    }
    if (rc) goto done;
    if (!activity) {
        unsigned K = fit->params[4];
        if (fit->param_size < 2 * K) goto done;
        for (size_t i = 0; i < n; ++i) {
            size_t offset = fit->param_size - 2 * K + 2 * classes[i];
            int16_t bias = (int16_t)(fit->params[offset] | (unsigned)fit->params[offset + 1] << 8);
            fit->base[i] = (uint16_t)(fit->base[i] - bias);
        }
        /* This fit is evaluated only with its own input predictor, so the
         * feature substitution term is exactly zero modulo 2^16. */
    }
    ok = 1;
done:
    free(feature);
    free(classes);
    if (!ok) vc_ng_fit_free(fit);
    return ok;
}
