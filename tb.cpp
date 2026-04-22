#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ap_fixed.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

typedef ap_fixed<16, 8>  neuron_act_t;
typedef ap_fixed<32, 16> accum_t;
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

static const char* CLASS_NAMES[NUM_CLASSES] = {"cyst", "normal", "stone", "tumor"};

void medivision_top(neuron_act_t input[IMG_H][IMG_W][IMG_C],output_t output[NUM_CLASSES]);

static int load_image_to_input(const char* filepath, neuron_act_t img[IMG_H][IMG_W][IMG_C])
{ 
    int src_w, src_h, src_channels;
    unsigned char* raw = stbi_load(filepath, &src_w, &src_h, &src_channels, 3);
 
    if (raw == NULL) {
        printf("[ERROR] stbi_load failed for: %s\n", filepath);
        printf("Reason: %s\n", stbi_failure_reason());
        return -1;
    }

 
    for (int h = 0; h < IMG_H; h++) 
    {
        for (int w = 0; w < IMG_W; w++) 
        {
            int base = (h * IMG_W + w) * 3;
 
            
            unsigned char r = raw[base + 0];
            unsigned char g = raw[base + 1];
            unsigned char b = raw[base + 2];
 
            img[h][w][0] = ((float)b / 255.0f); 
            img[h][w][1] = ((float)g / 255.0f);  
            img[h][w][2] = ((float)r / 255.0f); 
        }
    }
 
    stbi_image_free(raw);
    return 0;
}
 

static int run_image_test(const char* filepath, int class_index,neuron_act_t test_input[IMG_H][IMG_W][IMG_C],output_t test_output[NUM_CLASSES])
{
    printf("Image test: %s\n", filepath);

    printf("Expected class: %s\n", CLASS_NAMES[class_index]);
 
    if (load_image_to_input(filepath, test_input) != 0) 
    {
        printf("[SKIP] Could not load image.\n");
        return 0;
    }
 

    medivision_top(test_input, test_output);
 

    printf("\n  Output probabilities:\n");
    int pred_class = 0;
    float max_p    = (float)test_output[0];
    for (int i = 0; i < NUM_CLASSES; i++) 
    {
        float p = (float)test_output[i];
        printf("    [%d] %-8s : %.4f", i, CLASS_NAMES[i], p);
        if (i == class_index) printf("  <- expected");
        if (p > max_p) { max_p = p; pred_class = i; }
        printf("\n");
    }
    printf("  => Predicted: %s (p=%.4f)\n", CLASS_NAMES[pred_class], max_p);
    
   int mistake = 0;

    if (class_index >= 0) {
        if (pred_class == class_index)
        {
            printf("  Classification: [CORRECT]\n");
            mistake = 0;
        }
        else
        {
            printf("  Classification: [WRONG] (expected %s)\n", CLASS_NAMES[class_index]);
            mistake =  1;
        }
    }
    return mistake;
}
 
// -----------------------------------------------------------------------
// Main testbench
// -----------------------------------------------------------------------
int main()
{
 
    static neuron_act_t test_input[IMG_H][IMG_W][IMG_C];
    static output_t test_output[NUM_CLASSES];
 
    int failures = 0;
 
    printf("\n--- Real Image Tests ---\n\n");
 
    failures += run_image_test("C:/Users/arafa/Xilinx_projects/Project/Project/test/resized64/Cyst.png",  0, test_input, test_output);
    failures += run_image_test("C:/Users/arafa/Xilinx_projects/Project/Project/test/resized64/Normal.png", 1, test_input, test_output);
    failures += run_image_test("C:/Users/arafa/Xilinx_projects/Project/Project/test/resized64/Stone.png",  2, test_input, test_output);
    failures += run_image_test("C:/Users/arafa/Xilinx_projects/Project/Project/test/resized64/Tumor.png",  3, test_input, test_output);
 
    printf("\n=== Testbench complete: %d failure(s) ===\n", failures);
    return failures;
}