/* Earshot Lite: int8 keyword-spotting inference in plain C11.
 *
 * No libraries, no malloc. The model (weights, quantization parameters, layer
 * table) lives in a generated earshot_model.c; the runtime in earshot.c is
 * model-agnostic for the conv / depthwise conv / avgpool / fc layer set.
 *
 * Numerics follow earshot/quant/quantize.py::forward_int8 bit-for-bit.
 */
#ifndef EARSHOT_H
#define EARSHOT_H

#include <stddef.h>
#include <stdint.h>

/* Scratch sizes. Override with -D if you export a bigger model;
 * earshot_model.c static-asserts that its model fits. */
#ifndef ES_MAX_ACTIVATION
#define ES_MAX_ACTIVATION 32000 /* elements in the largest activation tensor */
#endif
#ifndef ES_MAX_INPUT
#define ES_MAX_INPUT 2048 /* in_frames * in_mels */
#endif
#ifndef ES_MAX_CHANNELS
#define ES_MAX_CHANNELS 64
#endif

typedef enum { ES_CONV, ES_DWCONV, ES_AVGPOOL, ES_FC } es_op;

typedef struct {
    es_op op;
    int in_h, in_w, in_c;  /* fc / avgpool output: out_c only */
    int out_h, out_w, out_c;
    int k_h, k_w, stride_h, stride_w, pad_top, pad_left;
    int32_t in_zp, out_zp, act_min;
    /* conv: [k_h][k_w][in_c][out_c]  dwconv: [k_h][k_w][c]  fc: [out_c][in_c] */
    const int8_t *weights;
    const int32_t *bias;  /* NULL for avgpool */
    const int32_t *mult;  /* per output channel */
    const int32_t *shift; /* per output channel */
} es_layer;

typedef struct {
    int num_layers;
    const es_layer *layers;
    int in_frames, in_mels, num_classes;
    const char *const *class_names;
    const float *norm_mean, *norm_std; /* per mel bin */
    float input_scale;
    int32_t input_zp;
    float output_scale;
    int32_t output_zp;
} es_model;

extern const es_model earshot_model;

/* Raw log-mel [in_frames][in_mels] -> normalized, quantized int8 input. */
void earshot_quantize_input(const float *logmel, int8_t *out);

/* Run the network on int8 input; returns a pointer to num_classes int8 logits
 * in an internal buffer (valid until the next call). Not reentrant. */
const int8_t *earshot_run(const int8_t *input);

/* Convenience: quantize + run + argmax. Copies logits to logits_out if non-NULL. */
int earshot_predict(const float *logmel, int8_t *logits_out);

#endif
