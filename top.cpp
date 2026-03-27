#include "ap_fixed.h"
#include "weights_bias.h"

typedef ap_fixed<16, 4>  neuron_act_t;
typedef ap_fixed<32, 21> accum_t;
typedef ap_fixed<16, 8>  output_t;

static const int NUM_CLASSES = 4;

// Input image: 64 x 64 x 3
static const int IMG_H   = 64;
static const int IMG_W   = 64;
static const int IMG_C   = 3;

// Conv1: 16 filters, 3x3, valid padding  -> 62 x 62 x 16
static const int CONV1_NUM_FILTERS = 16;
static const int CONV1_KERNAL_H      = 3;
static const int CONV1_KERNAL_W      = 3;
static const int CONV1_OUTPUT_H      = IMG_H   - CONV1_KERNAL_H + 1;   // 62
static const int CONV1_OUTPUT_W      = IMG_W   - CONV1_KERNAL_W + 1;   // 62

// MaxPool1: 2x2 stride 2  -> 31 x 31 x 16
static const int POOL1_OUTPUT_H = CONV1_OUTPUT_H / 2;   // 31
static const int POOL1_OUTPUT_W = CONV1_OUTPUT_W / 2;   // 31

// Conv2: 32 filters, 3x3, valid padding  -> 29 x 29 x 32
static const int CONV2_NUM_FILTERS = 32;
static const int CONV2_KERNAL_H      = 3;
static const int CONV2_KERNAL_W      = 3;
static const int CONV2_OUTPUT_H      = POOL1_OUTPUT_H - CONV2_KERNAL_H + 1;  // 29
static const int CONV2_OUTPUT_W      = POOL1_OUTPUT_W - CONV2_KERNAL_W + 1;  // 29

// MaxPool2: 2x2 stride 2  -> 14 x 14 x 32
static const int POOL2_OUTPUT_H = CONV2_OUTPUT_H / 2;   // 14
static const int POOL2_OUTPUT_W = CONV2_OUTPUT_W / 2;   // 14

// Flatten -> 14*14*32 = 6272
static const int FLAT_SIZE = POOL2_OUTPUT_H * POOL2_OUTPUT_W * CONV2_NUM_FILTERS;  // 6272

static neuron_act_t relu(accum_t x) {
#pragma HLS INLINE
    return (x > 0) ? (neuron_act_t) x : (neuron_act_t) 0;
}


static void conv1_layer(neuron_act_t input_fm[IMG_H][IMG_W][IMG_C], neuron_act_t output_fm[CONV1_OUTPUT_H][CONV1_OUTPUT_W][CONV1_NUM_FILTERS])
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

static void pool1_layer(neuron_act_t input_fm[CONV1_OUTPUT_H][CONV1_OUTPUT_W][CONV1_NUM_FILTERS],neuron_act_t output_fm[POOL1_OUTPUT_H][POOL1_OUTPUT_W][CONV1_NUM_FILTERS])
{
#pragma HLS INLINE off
    POOL1_OH_LOOP: for (int oh = 0; oh < POOL1_OUTPUT_H; oh++) 
    {
#pragma HLS PIPELINE II=1
        POOL1_OW_LOOP: for (int ow = 0; ow < POOL1_OUTPUT_W; ow++) 
        {
            POOL1_F_LOOP: for (int f = 0; f < CONV1_NUM_FILTERS; f++) 
            {
                neuron_act_t max = input_fm[oh*2][ow*2][f];
                if (input_fm[oh*2  ][ow*2+1][f] > max) 
                    max = input_fm[oh*2  ][ow*2+1][f];
                if (input_fm[oh*2+1][ow*2  ][f] > max) 
                    max = input_fm[oh*2+1][ow*2  ][f];
                if (input_fm[oh*2+1][ow*2+1][f] > max) 
                    max = input_fm[oh*2+1][ow*2+1][f];
                output_fm[oh][ow][f] = max;
            }
        }
    }
}

static void conv2_layer(neuron_act_t input_fm[POOL1_OUTPUT_H][POOL1_OUTPUT_W][CONV1_NUM_FILTERS],neuron_act_t output_fm[CONV2_OUTPUT_H][CONV2_OUTPUT_W][CONV2_NUM_FILTERS])
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

static void pool2_layer(neuron_act_t input_fm [CONV2_OUTPUT_H][CONV2_OUTPUT_W][CONV2_NUM_FILTERS],neuron_act_t output_fm[POOL2_OUTPUT_H][POOL2_OUTPUT_W][CONV2_NUM_FILTERS])
{
#pragma HLS INLINE off
    POOL2_OH_LOOP: for (int oh = 0; oh < POOL2_OUTPUT_H; oh++) 
    {
#pragma HLS PIPELINE II=1
        POOL2_OW_LOOP: for (int ow = 0; ow < POOL2_OUTPUT_W; ow++) 
        {
            POOL2_F_LOOP: for (int f = 0; f < CONV2_NUM_FILTERS; f++) 
            {
                neuron_act_t max = input_fm[oh*2][ow*2][f];
                if (input_fm[oh*2  ][ow*2+1][f] > max) max = input_fm[oh*2  ][ow*2+1][f];
                if (input_fm[oh*2+1][ow*2  ][f] > max) max = input_fm[oh*2+1][ow*2  ][f];
                if (input_fm[oh*2+1][ow*2+1][f] > max) max = input_fm[oh*2+1][ow*2+1][f];
                output_fm[oh][ow][f] = max;
            }
        }
    }
}

static void flatten_layer(neuron_act_t input_fm[POOL2_OUTPUT_H][POOL2_OUTPUT_W][CONV2_NUM_FILTERS],neuron_act_t flat[FLAT_SIZE])
{
#pragma HLS INLINE off
    int i = 0;
    FLAT_H: for (int h = 0; h < POOL2_OUTPUT_H; h++) {
        FLAT_W: for (int w = 0; w < POOL2_OUTPUT_W; w++) {
            FLAT_C: for (int c = 0; c < CONV2_NUM_FILTERS; c++) {
#pragma HLS PIPELINE II=1
                flat[i++] = input_fm[h][w][c];
            }
        }
    }
}


void medivision_top(neuron_act_t input[IMG_H][IMG_W][IMG_C],output_t output[NUM_CLASSES])
{


    // Intermediate feature maps
    static neuron_act_t conv1_out[CONV1_OUTPUT_H][CONV1_OUTPUT_W][CONV1_NUM_FILTERS];
    static neuron_act_t pool1_out[POOL1_OUTPUT_H][POOL1_OUTPUT_W][CONV1_NUM_FILTERS];
    static neuron_act_t conv2_out[CONV2_OUTPUT_H][CONV2_OUTPUT_W][CONV2_NUM_FILTERS];
    static neuron_act_t pool2_out[POOL2_OUTPUT_H][POOL2_OUTPUT_W][CONV2_NUM_FILTERS];
    static neuron_act_t flat_out [FLAT_SIZE];


    #pragma HLS ARRAY_PARTITION variable=conv1_out cyclic factor=16 dim=3
    #pragma HLS ARRAY_PARTITION variable=pool1_out cyclic factor=16 dim=3
    #pragma HLS ARRAY_PARTITION variable=conv2_out cyclic factor=32 dim=3
    #pragma HLS ARRAY_PARTITION variable=pool2_out cyclic factor=32 dim=3


    
    conv1_layer(input, conv1_out);
    pool1_layer(conv1_out, pool1_out);
    conv2_layer(pool1_out, conv2_out);
    pool2_layer(conv2_out, pool2_out);
    flatten_layer(pool2_out, flat_out);
}
