#include "earshot.h"

#include <math.h>
#include <string.h>

static int8_t buf_a[ES_MAX_ACTIVATION];
static int8_t buf_b[ES_MAX_ACTIVATION];

/* (acc * mult + 2^(shift-1)) >> shift, i.e. acc * mult / 2^shift rounded half up.
 * Relies on arithmetic right shift of negative int64 (true for gcc/clang/msvc). */
static inline int8_t requant(int32_t acc, int32_t mult, int32_t shift, int32_t zp, int32_t act_min) {
    int64_t r = ((int64_t)acc * mult + ((int64_t)1 << (shift - 1))) >> shift;
    r += zp;
    if (r < act_min) r = act_min;
    if (r > 127) r = 127;
    return (int8_t)r;
}

/* Accumulates all output channels of a pixel at once: the inner loop runs over
 * contiguous output channels (broadcast one input, multiply-add a weight row),
 * which vectorizes without horizontal reductions. */
static void conv(const es_layer *L, const int8_t *in, int8_t *out) {
    const int cin = L->in_c, cout = L->out_c;
    int32_t acc[ES_MAX_CHANNELS];
    for (int oy = 0; oy < L->out_h; oy++) {
        for (int ox = 0; ox < L->out_w; ox++) {
            for (int oc = 0; oc < cout; oc++) acc[oc] = L->bias[oc];
            for (int ky = 0; ky < L->k_h; ky++) {
                const int iy = oy * L->stride_h - L->pad_top + ky;
                if (iy < 0 || iy >= L->in_h) continue; /* padding == zero point -> contributes 0 */
                for (int kx = 0; kx < L->k_w; kx++) {
                    const int ix = ox * L->stride_w - L->pad_left + kx;
                    if (ix < 0 || ix >= L->in_w) continue;
                    const int8_t *x = in + (iy * L->in_w + ix) * cin;
                    const int8_t *w = L->weights + (ky * L->k_w + kx) * cin * cout;
                    for (int ic = 0; ic < cin; ic++) {
                        const int32_t xv = x[ic] - L->in_zp;
                        const int8_t *wr = w + ic * cout;
                        for (int oc = 0; oc < cout; oc++) acc[oc] += wr[oc] * xv;
                    }
                }
            }
            int8_t *o = out + (oy * L->out_w + ox) * cout;
            for (int oc = 0; oc < cout; oc++) o[oc] = requant(acc[oc], L->mult[oc], L->shift[oc], L->out_zp, L->act_min);
        }
    }
}

static void dwconv(const es_layer *L, const int8_t *in, int8_t *out) {
    const int c = L->out_c;
    int32_t acc[ES_MAX_CHANNELS];
    for (int oy = 0; oy < L->out_h; oy++) {
        for (int ox = 0; ox < L->out_w; ox++) {
            for (int ch = 0; ch < c; ch++) acc[ch] = L->bias[ch];
            for (int ky = 0; ky < L->k_h; ky++) {
                const int iy = oy * L->stride_h - L->pad_top + ky;
                if (iy < 0 || iy >= L->in_h) continue;
                for (int kx = 0; kx < L->k_w; kx++) {
                    const int ix = ox * L->stride_w - L->pad_left + kx;
                    if (ix < 0 || ix >= L->in_w) continue;
                    const int8_t *x = in + (iy * L->in_w + ix) * c;
                    const int8_t *w = L->weights + (ky * L->k_w + kx) * c;
                    for (int ch = 0; ch < c; ch++) acc[ch] += w[ch] * (x[ch] - L->in_zp);
                }
            }
            int8_t *o = out + (oy * L->out_w + ox) * c;
            for (int ch = 0; ch < c; ch++) o[ch] = requant(acc[ch], L->mult[ch], L->shift[ch], L->out_zp, L->act_min);
        }
    }
}

static void avgpool(const es_layer *L, const int8_t *in, int8_t *out) {
    const int n = L->in_h * L->in_w, c = L->in_c;
    int32_t acc[ES_MAX_CHANNELS] = {0};
    for (int i = 0; i < n; i++)
        for (int ch = 0; ch < c; ch++) acc[ch] += in[i * c + ch] - L->in_zp;
    for (int ch = 0; ch < c; ch++) out[ch] = requant(acc[ch], L->mult[ch], L->shift[ch], L->out_zp, L->act_min);
}

static void fc(const es_layer *L, const int8_t *in, int8_t *out) {
    for (int o = 0; o < L->out_c; o++) {
        const int8_t *w = L->weights + o * L->in_c;
        int32_t acc = L->bias[o];
        for (int i = 0; i < L->in_c; i++) acc += w[i] * (in[i] - L->in_zp);
        out[o] = requant(acc, L->mult[o], L->shift[o], L->out_zp, L->act_min);
    }
}

void earshot_quantize_input(const float *logmel, int8_t *out) {
    const es_model *m = &earshot_model;
    for (int t = 0; t < m->in_frames; t++) {
        for (int f = 0; f < m->in_mels; f++) {
            const float v = (logmel[t * m->in_mels + f] - m->norm_mean[f]) / m->norm_std[f];
            long q = lrintf(v / m->input_scale) + m->input_zp; /* round half to even, like np.rint */
            out[t * m->in_mels + f] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
        }
    }
}

const int8_t *earshot_run(const int8_t *input) {
    const es_model *m = &earshot_model;
    const int8_t *in = input;
    int8_t *out = buf_a;
    for (int i = 0; i < m->num_layers; i++) {
        const es_layer *L = &m->layers[i];
        switch (L->op) {
            case ES_CONV: conv(L, in, out); break;
            case ES_DWCONV: dwconv(L, in, out); break;
            case ES_AVGPOOL: avgpool(L, in, out); break;
            case ES_FC: fc(L, in, out); break;
        }
        in = out;
        out = (out == buf_a) ? buf_b : buf_a;
    }
    return in;
}

int earshot_predict(const float *logmel, int8_t *logits_out) {
    const es_model *m = &earshot_model;
    static int8_t q[ES_MAX_INPUT];
    earshot_quantize_input(logmel, q);
    const int8_t *logits = earshot_run(q);
    if (logits_out) memcpy(logits_out, logits, (size_t)m->num_classes);
    int best = 0;
    for (int i = 1; i < m->num_classes; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}
