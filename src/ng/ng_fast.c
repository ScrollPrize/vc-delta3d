/* Fast preset: the original distributed screen, one full-chunk fit, and DP64
 * entropy thresholds. Candidate ordering and strict comparisons are deliberate:
 * they preserve the existing format bytes, including tied screening choices. */
#include "ng_fit.h"
#include "ng_frame.h"
#include "ng_predict.h"
#include "ng_screen.h"
#include "ng_thresholds.h"
#include "residual_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

typedef struct {
    uint8_t *empty, *mask;
    size_t count, blocks;
} occupancy;

static int scan(occupancy *o, const uint8_t *src, unsigned d)
{
    o->blocks = (size_t)(d / 4) * (d / 4) * (d / 4);
    o->empty = malloc(o->blocks);
    if (!o->empty) return 0;
    o->count = vc_ng_zero_scan(src, d, 4, o->empty);
    if (o->count) {
        o->mask = malloc((size_t)d * d * d);
        if (!o->mask) return 0;
        vc_ng_zero_mask(d, 4, o->empty, o->mask);
    }
    return 1;
}

static int statistics(const uint8_t *symbols, const uint8_t *mask, size_t n,
                      uint16_t *activity, uint8_t *classes, uint16_t thresholds[15],
                      uint32_t counts[16][256], uint32_t histogram[2560])
{
    if (ng_activity_symbol_histogram(symbols, activity, histogram)) return 0;
    if (mask) {
        memset(histogram, 0, 2560 * sizeof(*histogram));
        for (size_t i = 0; i < n; ++i) if (mask[i]) ++histogram[activity[i]];
    }
    if (!vc_ng_quantiles(histogram, 16, thresholds) ||
        !vc_ng_classify(activity, n, thresholds, 16, classes)) return 0;
    memset(counts, 0, 16 * 256 * sizeof(uint32_t));
    for (size_t i = 0; i < n; ++i)
        if (!mask || mask[i]) ++counts[classes[i]][symbols[i]];
    return 1;
}

static double uniform_cost(uint32_t counts[16][256], unsigned scale, size_t blocks)
{
    uint32_t selectors[256] = {0};
    selectors[0] = (uint32_t)(blocks * scale);
    double cost = vc_ng_histogram_cost(selectors, 12, 1);
    for (unsigned k = 0; k < 16; ++k) {
        uint32_t scaled[256];
        for (unsigned j = 0; j < 256; ++j) scaled[j] = counts[k][j] * scale;
        cost += vc_ng_histogram_cost(scaled, 12, 1);
    }
    return cost / 8.;
}

static double metadata_cost(const vc_ng_fit *fit, const uint8_t *empty, size_t empty_size)
{
    size_t size = fit ? 2 * fit->param_size + 8 : 2 * empty_size;
    uint8_t *data = malloc(size), *compressed = malloc(ZSTD_compressBound(size));
    if (!data || !compressed) {free(data);free(compressed);return -1;}
    if (fit) {
        memcpy(data, fit->params, fit->param_size);
        for (unsigned j = 0; j < 2; ++j)
            for (unsigned b = 0; b < 4; ++b)
                data[fit->param_size + 4 * j + b] = (uint8_t)((uint32_t)fit->weights[j] >> (8 * b));
        memcpy(data + fit->param_size + 8, fit->params, fit->param_size);
    } else {
        memcpy(data, empty, empty_size);
        memcpy(data + empty_size, empty, empty_size);
    }
    size_t bytes = ZSTD_compress(compressed, ZSTD_compressBound(size), data, size, 3);
    free(data);free(compressed);
    return ZSTD_isError(bytes) ? -1 : (double)bytes + 12.;
}

int vc_ng_fast_encode(const uint8_t *src, unsigned d, uint8_t *out, size_t cap, size_t *written)
{
    if (!src || !out || !written || !ng_geometry_valid(d,d,d)) return -1;
    const ng_geometry saved_geometry = ng_current_geometry;
    const int saved_phase = ng_tap_fit_phase;
    const unsigned saved_edge = ng_choice_edge;
    const size_t full_n = (size_t)d * d * d;
    const unsigned sd = d > 32 ? 32 : d;
    const size_t n = (size_t)sd * sd * sd;
    const unsigned scale = (unsigned)(full_n / n);
    occupancy full = {0}, screen = {0};
    vc_ng_fit fits[2] = {{0}}, final_fit = {0};
    uint8_t *proxy = NULL, *candidate[12] = {0}, *classes = NULL, *symbols = NULL;
    uint8_t *selected = NULL, *active_symbols = NULL, *frame = NULL;
    uint16_t *activity = NULL, *active_activity = NULL;
    void *session = NULL;
    size_t frame_size = 0, empty_size = 0;
    uint8_t empty[65536];
    uint32_t counts[16][256], histogram[2560];
    uint16_t thresholds[15], final_thresholds[15];
    double screening[12], costs[5];
    unsigned top[2] = {0,0}, options[5], option_count = 1, chosen = 0;
    int rc = -1, activity_ready = 0;
    ng_current_geometry = (ng_geometry){(int)d,(int)d,(int)d};
    ng_choice_edge = 4;
    ng_tap_fit_phase = 1;
    if (!scan(&full,src,d)) goto done;
    selected = calloc(full.blocks,1);
    if (!selected) goto done;
    if (full.count == full.blocks) {
        vc_ng_choice zero = {12,255,0,{0,0}};
        uint16_t t[15] = {0};
        if (!vc_ng_frame_encode(d,4,1,NULL,0,&zero,1,selected,NULL,t,16,NULL,&frame,&frame_size)) goto done;
        goto finish;
    }
    const uint8_t *sample = src;
    if (d > 32) {
        proxy = malloc(n);
        if (!proxy || vc_ng_screen_proxy(src,d,4,full.count?full.empty:NULL,proxy)) goto done;
        sample = proxy;
    }
    ng_current_geometry = (ng_geometry){(int)sd,(int)sd,(int)sd};
    if (full.count && !scan(&screen,sample,sd)) goto done;
    classes = malloc(full_n);
    activity = malloc(full_n * sizeof(*activity));
    if (!classes || !activity) goto done;
    for (unsigned m = 0; m < 12; ++m) {
        candidate[m] = malloc(n);
        if (!candidate[m]) goto done;
    }
    vc_ng_fixed_candidates(sample,candidate,sd,4095);
    for (unsigned m = 0; m < 12; ++m) {
        if (!statistics(candidate[m],screen.mask,n,activity,classes,thresholds,counts,histogram)) goto done;
        double cost = 0;
        for (unsigned k = 0; k < 16; ++k) cost += vc_ng_histogram_cost(counts[k],12,1);
        screening[m] = cost;
    }
    /* Stable ascending predictor ID resolves equal histogram costs. */
    for (unsigned m = 1; m < 12; ++m) if (screening[m] < screening[top[0]]) top[0] = m;
    top[1] = top[0] == 0 ? 1 : 0;
    for (unsigned m = 0; m < 12; ++m)
        if (m != top[0] && screening[m] < screening[top[1]]) top[1] = m;
    int cfg[6] = {3,3,1,16,40,d>32?4:128};
    session = ng_fit_session_create(1);
    if (!session) goto done;
    for (unsigned k = 0; k < 2; ++k)
        if (!vc_ng_fit_input(&fits[k],sample,cfg,candidate[top[k]],screen.mask,
                            screen.count?NULL:session,NULL,NULL)) goto done;
    ng_fit_session_destroy(session);session = NULL;
    options[0] = 0;
    for (unsigned k = 0; k < 2; ++k) {
        unsigned ids[2] = {top[k],12+12*k+top[k]};
        for (unsigned j = 0; j < 2; ++j) {
            unsigned p = 0;
            while (p < option_count && options[p] != ids[j]) ++p;
            if (p == option_count) options[option_count++] = ids[j];
        }
    }
    if (ng_bank_empty_model(empty,sizeof(empty),&empty_size)) goto done;
    for (unsigned j = 0; j < option_count; ++j) {
        unsigned id = options[j];
        const vc_ng_fit *fit = id >= 12 ? &fits[(id-12)/12] : NULL;
        const uint8_t *s = fit ? fit->symbols : candidate[id];
        if (!statistics(s,screen.mask,n,activity,classes,thresholds,counts,histogram)) goto done;
        double metadata = metadata_cost(fit,empty,empty_size);
        if (metadata < 0) goto done;
        costs[j] = uniform_cost(counts,scale,n/64) + metadata;
        if (costs[j] < costs[chosen]) chosen = j;
    }
    chosen = options[chosen];
    unsigned predictor = chosen % 12;
    const int learned = chosen >= 12;
    if (d > 32) {
        ng_current_geometry = (ng_geometry){(int)d,(int)d,(int)d};
        ng_tap_fit_phase = 2;
        for (unsigned m = 0; m < 12; ++m) {free(candidate[m]);candidate[m]=NULL;}
        unsigned fixed_mask = (1u<<predictor) | (full.count?1u:0u);
        for (unsigned m = 0; m < 12; ++m) if (fixed_mask & (1u<<m)) {
            candidate[m] = malloc(full_n);
            if (!candidate[m]) goto done;
        }
        vc_ng_fixed_candidates(src,candidate,d,fixed_mask);
        if (learned) {
            cfg[5] = 128;
            if (!vc_ng_fit_input(&final_fit,src,cfg,candidate[predictor],full.mask,NULL,
                                full.count?NULL:activity,full.count?NULL:histogram)) goto done;
            activity_ready = !full.count;
        }
        if (ng_bank_empty_model(empty,sizeof(empty),&empty_size)) goto done;
    } else if (learned) {
        unsigned k = (chosen-12)/12;
        final_fit = fits[k];memset(&fits[k],0,sizeof(fits[k]));
    }
    symbols = malloc(full_n);
    if (!symbols) goto done;
    memcpy(symbols,learned?final_fit.symbols:candidate[predictor],full_n);
    if (!activity_ready && ng_activity_symbol_histogram(symbols,activity,histogram)) goto done;
    unsigned seed = learned ? 12+predictor : predictor;
    memset(selected,seed,full.blocks);
    if (full.count) {
        for (size_t b = 0; b < full.blocks; ++b) if (full.empty[b]) selected[b] = 0;
        if (learned) {
            const uint8_t *params[12];size_t sizes[12];const uint16_t *bases[12];
            uint8_t *outputs[24] = {0};
            for (unsigned m = 0; m < 12; ++m) {
                params[m]=final_fit.params;sizes[m]=final_fit.param_size;bases[m]=final_fit.base;
                outputs[m]=candidate[m];
            }
            outputs[12+predictor]=final_fit.symbols;
            if (ng_bank_selected_symbols(12,params,sizes,bases,selected,src,outputs,symbols,activity)) goto done;
        } else {
            for (size_t i = 0; i < full_n; ++i) if (!full.mask[i]) symbols[i]=0;
            if (ng_activity_symbol_values(symbols,activity)) goto done;
        }
    }
    size_t active_n = full_n;
    const uint8_t *ss = symbols;const uint16_t *aa = activity;
    if (full.count) {
        active_n = full_n - full.count*64;
        active_symbols = malloc(active_n);
        active_activity = malloc(active_n*sizeof(*active_activity));
        if (!active_symbols || !active_activity) goto done;
        memset(histogram,0,sizeof(histogram));
        size_t j=0;
        for (size_t i=0;i<full_n;++i) if (full.mask[i]) {
            active_symbols[j]=symbols[i];active_activity[j++]=activity[i];++histogram[activity[i]];
        }
        ss=active_symbols;aa=active_activity;
    }
    if (!vc_ng_quantiles(histogram,16,thresholds) ||
        !vc_ng_refine_thresholds(ss,aa,active_n,thresholds,16,NULL,NULL,0,1,1,64,1,final_thresholds) ||
        !vc_ng_classify(activity,full_n,final_thresholds,16,classes)) goto done;
    vc_ng_model model = {learned?final_fit.params:empty,learned?final_fit.param_size:empty_size};
    vc_ng_choice menu[3];unsigned remap[24],count=0;
    for (unsigned j=0;j<24;++j) remap[j]=255;
    for (size_t b=0;b<full.blocks;++b) {
        unsigned id=selected[b];
        if (remap[id]==255) {
            unsigned m=id>=12?0:255;
            menu[count]=(vc_ng_choice){id%12,m,0,{0,0}};
            if (m!=255 && id%12) memcpy(menu[count].weights,final_fit.weights,sizeof(final_fit.weights));
            remap[id]=count++;
        }
        selected[b]=(uint8_t)remap[id];
    }
    if (full.count) {
        menu[count]=(vc_ng_choice){12,255,1,{0,0}};
        for (size_t b=0;b<full.blocks;++b) if (full.empty[b]) selected[b]=(uint8_t)count;
        ++count;
    }
    if (!vc_ng_frame_encode(d,4,1,&model,1,menu,count,selected,symbols,final_thresholds,16,classes,&frame,&frame_size)) goto done;
finish:
    if (frame_size>cap) goto done;
    memcpy(out,frame,frame_size);*written=frame_size;rc=0;
done:
    if (session) ng_fit_session_destroy(session);
    for (unsigned k=0;k<2;++k) vc_ng_fit_free(&fits[k]);
    vc_ng_fit_free(&final_fit);
    for (unsigned m=0;m<12;++m) free(candidate[m]);
    free(full.empty);free(full.mask);free(screen.empty);free(screen.mask);
    free(proxy);free(classes);free(symbols);free(activity);free(selected);
    free(active_symbols);free(active_activity);free(frame);
    ng_current_geometry=saved_geometry;ng_tap_fit_phase=saved_phase;ng_choice_edge=saved_edge;
    return rc;
}
