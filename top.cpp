#include "ap_fixed.h"
#include "weights_bias.h"
#include <math.h>

typedef ap_fixed<16, 8>  neuron_act_t;   
typedef ap_fixed<32, 16> accum_t;       
typedef ap_fixed<16, 8>  output_t;       

static const int NUM_CLASSES = 4;

static const int IMG_H = 64;
static const int IMG_W = 64;
static const int IMG_C = 3;

static const int CONV1_NUM_FILTERS = 16;
static const int CONV1_KERNAL_H    = 3;
static const int CONV1_KERNAL_W    = 3;
static const int CONV1_OUTPUT_H    = IMG_H - CONV1_KERNAL_H + 1;   // 62
static const int CONV1_OUTPUT_W    = IMG_W - CONV1_KERNAL_W + 1;   // 62

static const int POOL1_OUTPUT_H = CONV1_OUTPUT_H / 2;   // 31
static const int POOL1_OUTPUT_W = CONV1_OUTPUT_W / 2;   // 31

static const int CONV2_NUM_FILTERS = 32;
static const int CONV2_KERNAL_H    = 3;
static const int CONV2_KERNAL_W    = 3;
static const int CONV2_OUTPUT_H    = POOL1_OUTPUT_H - CONV2_KERNAL_H + 1;  // 29
static const int CONV2_OUTPUT_W    = POOL1_OUTPUT_W - CONV2_KERNAL_W + 1;  // 29

static const int POOL2_OUTPUT_H = CONV2_OUTPUT_H / 2;   // 14
static const int POOL2_OUTPUT_W = CONV2_OUTPUT_W / 2;   // 14

static const int FLAT_SIZE  = POOL2_OUTPUT_H * POOL2_OUTPUT_W * CONV2_NUM_FILTERS; // 6272
static const int LSTM_UNITS = 50;

// -----------------------------------------------------------------------
// Activation helpers
// -----------------------------------------------------------------------

static neuron_act_t relu(accum_t x) 
{
    #pragma HLS INLINE
    return (x > 0) ? (neuron_act_t)x : (neuron_act_t)0;
}

static float sigmoid_f(float x) 
{
    #pragma HLS INLINE
    return 1.0f / (1.0f + expf(-x));
}

static float tanh_f(float x) 
{
    #pragma HLS INLINE
    return tanhf(x);
}

// -----------------------------------------------------------------------
// conv1_layer
// -----------------------------------------------------------------------
static void conv1_layer(neuron_act_t input_fm [IMG_H][IMG_W][IMG_C], neuron_act_t output_fm[CONV1_OUTPUT_H][CONV1_OUTPUT_W][CONV1_NUM_FILTERS])
{
    #pragma HLS INLINE off
    CONV1_OH_LOOP: for (int oh = 0; oh < CONV1_OUTPUT_H; oh++) 
    {
        #pragma HLS PIPELINE II=1
        CONV1_OW_LOOP: for (int ow = 0; ow < CONV1_OUTPUT_W; ow++) 
        {
            CONV1_F_LOOP: for (int f = 0; f < CONV1_NUM_FILTERS; f++) 
            {
                accum_t acc = (accum_t) conv1_b[f];
                CONV1_KH_LOOP: for (int kh = 0; kh < CONV1_KERNAL_H; kh++) 
                {
                    CONV1_KW_LOOP: for (int kw = 0; kw < CONV1_KERNAL_W; kw++) 
                    {
                        CONV1_IC_LOOP: for (int ic = 0; ic < IMG_C; ic++) 
                        {
                            acc += (accum_t) input_fm[oh+kh][ow+kw][ic] * (accum_t) conv1_w[f][ic][kh][kw];
                        }
                    }
                }
                output_fm[oh][ow][f] = relu(acc);
            }
        }
    }
}

// -----------------------------------------------------------------------
// pool1_layer
// -----------------------------------------------------------------------
static void pool1_layer(neuron_act_t input_fm[CONV1_OUTPUT_H][CONV1_OUTPUT_W][CONV1_NUM_FILTERS], neuron_act_t output_fm[POOL1_OUTPUT_H][POOL1_OUTPUT_W][CONV1_NUM_FILTERS])
{
    #pragma HLS INLINE off
    POOL1_OH_LOOP: for (int oh = 0; oh < POOL1_OUTPUT_H; oh++) {
    #pragma HLS PIPELINE II=1
    POOL1_OW_LOOP: for (int ow = 0; ow < POOL1_OUTPUT_W; ow++) 
    {
        POOL1_F_LOOP: for (int f = 0; f < CONV1_NUM_FILTERS; f++) 
        {
            neuron_act_t mx = input_fm[oh*2][ow*2][f];
            if (input_fm[oh*2  ][ow*2+1][f] > mx) 
                mx = input_fm[oh*2  ][ow*2+1][f];
            if (input_fm[oh*2+1][ow*2  ][f] > mx) 
                mx = input_fm[oh*2+1][ow*2  ][f];
            if (input_fm[oh*2+1][ow*2+1][f] > mx) 
                mx = input_fm[oh*2+1][ow*2+1][f];
            output_fm[oh][ow][f] = mx;
            }
        }
    }
}

// -----------------------------------------------------------------------
// conv2_layer
// -----------------------------------------------------------------------
static void conv2_layer(neuron_act_t input_fm[POOL1_OUTPUT_H][POOL1_OUTPUT_W][CONV1_NUM_FILTERS], neuron_act_t output_fm[CONV2_OUTPUT_H][CONV2_OUTPUT_W][CONV2_NUM_FILTERS])
{
    #pragma HLS INLINE off
    CONV2_OUTPUT_H_LOOP: for (int oh = 0; oh < CONV2_OUTPUT_H; oh++) 
    {
        #pragma HLS PIPELINE II=1
        CONV2_OUTPUT_W_LOOP: for (int ow = 0; ow < CONV2_OUTPUT_W; ow++) 
        {
            CONV2_F_LOOP: for (int f = 0; f < CONV2_NUM_FILTERS; f++) 
            {
                accum_t acc = (accum_t) conv2_b[f];
                CONV2_KERNAL_H_LOOP: for (int kh = 0; kh < CONV2_KERNAL_H; kh++) 
                {
                    CONV2_KERNAL_W_LOOP: for (int kw = 0; kw < CONV2_KERNAL_W; kw++) 
                    {
                        CONV2_IC_LOOP: for (int ic = 0; ic < CONV1_NUM_FILTERS; ic++) 
                        {
                            acc += (accum_t) input_fm[oh+kh][ow+kw][ic] * (accum_t) conv2_w[f][ic][kh][kw];
                        }
                    }
                }
                output_fm[oh][ow][f] = relu(acc);
            }
        }
    }
}

// -----------------------------------------------------------------------
// pool2_layer
// -----------------------------------------------------------------------
static void pool2_layer(neuron_act_t input_fm[CONV2_OUTPUT_H][CONV2_OUTPUT_W][CONV2_NUM_FILTERS], neuron_act_t output_fm[POOL2_OUTPUT_H][POOL2_OUTPUT_W][CONV2_NUM_FILTERS])
{
    #pragma HLS INLINE off
    POOL2_OH_LOOP: for (int oh = 0; oh < POOL2_OUTPUT_H; oh++) 
    {
        #pragma HLS PIPELINE II=1
        POOL2_OW_LOOP: for (int ow = 0; ow < POOL2_OUTPUT_W; ow++) 
        {
            POOL2_F_LOOP: for (int f = 0; f < CONV2_NUM_FILTERS; f++) 
            {
                neuron_act_t mx = input_fm[oh*2][ow*2][f];
                if (input_fm[oh*2  ][ow*2+1][f] > mx) 
                    mx = input_fm[oh*2  ][ow*2+1][f];
                if (input_fm[oh*2+1][ow*2  ][f] > mx) 
                    mx = input_fm[oh*2+1][ow*2  ][f];
                if (input_fm[oh*2+1][ow*2+1][f] > mx) 
                    mx = input_fm[oh*2+1][ow*2+1][f];
                output_fm[oh][ow][f] = mx;
            }
        }
    }
}

// -----------------------------------------------------------------------
// flatten_layer
// -----------------------------------------------------------------------
static void flatten_layer(neuron_act_t input_fm[POOL2_OUTPUT_H][POOL2_OUTPUT_W][CONV2_NUM_FILTERS], float flat[FLAT_SIZE])
{
    #pragma HLS INLINE off
    int i = 0;
    FLAT_H: for (int h = 0; h < POOL2_OUTPUT_H; h++) 
    {
        FLAT_W: for (int w = 0; w < POOL2_OUTPUT_W; w++) 
        {
            FLAT_C: for (int c = 0; c < CONV2_NUM_FILTERS; c++) 
            {
                #pragma HLS PIPELINE II=1
                flat[i++] = (float)input_fm[h][w][c];
            }
        }
    }
}


static void lstm_layer(float flat_input[FLAT_SIZE], float h_prev[LSTM_UNITS], float c_prev[LSTM_UNITS], float h_out[LSTM_UNITS], float c_out[LSTM_UNITS])
{
    #pragma HLS INLINE off

    float raw_i[LSTM_UNITS];
    float raw_f[LSTM_UNITS];
    float raw_g[LSTM_UNITS];
    float raw_o[LSTM_UNITS];

    #pragma HLS ARRAY_PARTITION variable=raw_i complete
    #pragma HLS ARRAY_PARTITION variable=raw_f complete
    #pragma HLS ARRAY_PARTITION variable=raw_g complete
    #pragma HLS ARRAY_PARTITION variable=raw_o complete

    LSTM_INIT: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        raw_i[u] = (float) lstm_b[0 * LSTM_UNITS + u];
        raw_f[u] = (float) lstm_b[1 * LSTM_UNITS + u];
        raw_g[u] = (float) lstm_b[2 * LSTM_UNITS + u];
        raw_o[u] = (float) lstm_b[3 * LSTM_UNITS + u];
    }

    LSTM_WX_LOOP: for (int i = 0; i < FLAT_SIZE; i++) 
    {
        #pragma HLS PIPELINE II=1
        float xi = flat_input[i];
        LSTM_WX_U: for (int u = 0; u < LSTM_UNITS; u++) 
        {
            #pragma HLS UNROLL
            raw_i[u] += xi * (float) lstm_w[i][0 * LSTM_UNITS + u];
            raw_f[u] += xi * (float) lstm_w[i][1 * LSTM_UNITS + u];
            raw_g[u] += xi * (float) lstm_w[i][2 * LSTM_UNITS + u];
            raw_o[u] += xi * (float) lstm_w[i][3 * LSTM_UNITS + u];
        }
    }


    LSTM_UH_LOOP: for (int k = 0; k < LSTM_UNITS; k++) 
    {
        #pragma HLS PIPELINE II=1
        float hk = h_prev[k];
        LSTM_UH_U: for (int u = 0; u < LSTM_UNITS; u++) 
        {
            #pragma HLS UNROLL
            raw_i[u] += hk * (float) lstm_w_2[k][0 * LSTM_UNITS + u];
            raw_f[u] += hk * (float) lstm_w_2[k][1 * LSTM_UNITS + u];
            raw_g[u] += hk * (float) lstm_w_2[k][2 * LSTM_UNITS + u];
            raw_o[u] += hk * (float) lstm_w_2[k][3 * LSTM_UNITS + u];
        }
    }


    LSTM_OUT_LOOP: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        float i_gate = sigmoid_f(raw_i[u]);
        float f_gate = sigmoid_f(raw_f[u]);
        float g_gate = tanh_f(raw_g[u]);
        float o_gate = sigmoid_f(raw_o[u]);

        float c_new  = f_gate * c_prev[u] + i_gate * g_gate;
        c_out[u] = c_new;
        h_out[u] = o_gate * tanh_f(c_new);
    }
}


static void attention_layer(float h_in[LSTM_UNITS], float combined[LSTM_UNITS])
{
    #pragma HLS INLINE off

    float score[LSTM_UNITS];
    #pragma HLS ARRAY_PARTITION variable=score complete

    
    ATT_INIT: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        score[u] = (float) att_b[u];
    }


    ATT_SCORE_LOOP: for (int v = 0; v < LSTM_UNITS; v++) 
    {
        #pragma HLS PIPELINE II=1
        float hv = h_in[v];
        ATT_SCORE_U: for (int u = 0; u < LSTM_UNITS; u++) 
        {
            #pragma HLS UNROLL
            score[u] += hv * (float) att_w[v][u];
        }
    }


    float et[LSTM_UNITS];
    #pragma HLS ARRAY_PARTITION variable=et complete

    ATT_TANH: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        et[u] = tanh_f(score[u]);
    }


    float att_max = et[0];
    ATT_MAX: for (int u = 1; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        if (et[u] > att_max) 
            att_max = et[u];
    }

    float att_exp[LSTM_UNITS];
    float att_sum = 0.0f;
    #pragma HLS ARRAY_PARTITION variable=att_exp complete

    ATT_EXP: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        att_exp[u] = expf(et[u] - att_max);
        att_sum   += att_exp[u];
    }

    float att_weight[LSTM_UNITS];
    #pragma HLS ARRAY_PARTITION variable=att_weight complete

    ATT_NORM: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        att_weight[u] = att_exp[u] / att_sum;
    }


    ATT_OUT: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        combined[u] = h_in[u] * att_weight[u] + h_in[u];
    }
}

static void classifier_dense_layer(float input[LSTM_UNITS], output_t output[NUM_CLASSES])
{
    #pragma HLS INLINE off
    float logits[NUM_CLASSES];

    CLS_OUT: for (int j = 0; j < NUM_CLASSES; j++) 
    {
        #pragma HLS UNROLL
        float acc = (float) final_b[j];
        CLS_IN: for (int i = 0; i < LSTM_UNITS; i++) 
        {
            #pragma HLS UNROLL
            acc += input[i] * (float) final_w[i][j];
        }
        logits[j] = acc;
    }

    float max_val = logits[0];
    for (int j = 1; j < NUM_CLASSES; j++) 
    {
        #pragma HLS UNROLL
        if (logits[j] > max_val) 
            max_val = logits[j];
    }

    float sum_exp = 0.0f;
    float exps[NUM_CLASSES];

    SOFTMAX_EXP: for (int j = 0; j < NUM_CLASSES; j++) 
    {
        #pragma HLS UNROLL
        exps[j]  = expf(logits[j] - max_val);
        sum_exp += exps[j];
    }

    SOFTMAX_NORM: for (int j = 0; j < NUM_CLASSES; j++) 
    {
        #pragma HLS UNROLL
        output[j] = (output_t) (exps[j] / sum_exp);
    }
}

// -----------------------------------------------------------------------
// medivision_top  -- top-level function
// -----------------------------------------------------------------------
void medivision_top(neuron_act_t input[IMG_H][IMG_W][IMG_C], output_t output[NUM_CLASSES])
{

    static neuron_act_t conv1_out[CONV1_OUTPUT_H][CONV1_OUTPUT_W][CONV1_NUM_FILTERS];
    static neuron_act_t pool1_out[POOL1_OUTPUT_H][POOL1_OUTPUT_W][CONV1_NUM_FILTERS];
    static neuron_act_t conv2_out[CONV2_OUTPUT_H][CONV2_OUTPUT_W][CONV2_NUM_FILTERS];
    static neuron_act_t pool2_out[POOL2_OUTPUT_H][POOL2_OUTPUT_W][CONV2_NUM_FILTERS];

    static float flat_out[FLAT_SIZE];
    static float lstm_h_prev[LSTM_UNITS];
    static float lstm_c_prev[LSTM_UNITS];
    static float lstm_h_out[LSTM_UNITS];
    static float lstm_c_out[LSTM_UNITS];
    static float att_combined[LSTM_UNITS];

    #pragma HLS ARRAY_PARTITION variable=conv1_out cyclic factor=16 dim=3
    #pragma HLS ARRAY_PARTITION variable=pool1_out cyclic factor=16 dim=3
    #pragma HLS ARRAY_PARTITION variable=conv2_out cyclic factor=32 dim=3
    #pragma HLS ARRAY_PARTITION variable=pool2_out cyclic factor=32 dim=3
    #pragma HLS ARRAY_PARTITION variable=lstm_h_prev complete
    #pragma HLS ARRAY_PARTITION variable=lstm_c_prev complete
    #pragma HLS ARRAY_PARTITION variable=lstm_h_out complete
    #pragma HLS ARRAY_PARTITION variable=lstm_c_out complete
    #pragma HLS ARRAY_PARTITION variable=att_combined complete

    LSTM_RESET: for (int u = 0; u < LSTM_UNITS; u++) 
    {
        #pragma HLS UNROLL
        lstm_h_prev[u] = 0.0f;
        lstm_c_prev[u] = 0.0f;
    }

    conv1_layer(input, conv1_out);
    pool1_layer(conv1_out, pool1_out);
    conv2_layer(pool1_out, conv2_out);
    pool2_layer(conv2_out, pool2_out);
    flatten_layer(pool2_out, flat_out);
    lstm_layer(flat_out, lstm_h_prev, lstm_c_prev, lstm_h_out, lstm_c_out);
    attention_layer(lstm_h_out, att_combined);
    classifier_dense_layer(att_combined, output);
}