/*
 * Created on Fri Jan 5 10:28:48 2024
 *
 * Author: Thuong_Duong
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>

#define LABEL_LEN 10000

// Quantization parameters
typedef struct {
    float scale;
    int bit_width;
    int shift; // Shift value for requantization
} QuantParams;

// Quantize a floating-point value to an integer using symmetric quantization
int quantize(float value, QuantParams params) {
    float scaled_value = value / params.scale;
    int quantized_value = (int)round(scaled_value);
    int max_value = (1 << (params.bit_width - 1)) - 1;
    int min_value = - (1 << (params.bit_width - 1));
    if (quantized_value > max_value) {
        return max_value;
    } else if (quantized_value < min_value) {
        return min_value;
    } else {
        return quantized_value;
    }
}


// ReLu activation function for integer
void relu_int(int32_t *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) {
            x[i] = 0;
        }
    }
}

// Softmax activation function for integer
void softmax_int(int32_t *x, float *output, int size, QuantParams act_params) {
    int32_t max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(((float)(x[i] - max)) / (float)(1 << act_params.shift));
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}


void Prediction( float image[28][28], // Input image
    float w_conv1[6][1][1], // weight of convolution 1
    float w_conv2[16][6][5][5], // weight of convolution 2
    float w_fc1[120][400], // weight of fully connection 1
    float w_fc2[84][120], // weight of fully connection 2
    float w_fc3[10][84], // weight of fully connection 3
    float b_conv1[6], // bias of convolution 1
    float b_conv2[16], // bias of convolution 2
    float b_fc1[120], // bias of fully connection 1
    float b_fc2[84], // bias of fully connection 2
    float b_fc3[10], // bias of fully connection 3
    float probs[10], //probs of 10 classes through softmax activate function
    QuantParams weight_params, QuantParams act_params);

int main(int argc, char** argv){

    float w_conv1[6][1][1];
    float w_conv2[16][6][5][5];
    float w_fc1[120][400];
    float w_fc2[84][120];
    float w_fc3[10][84];
    float b_conv1[6];
    float b_conv2[16];
    float b_fc1[120];
    float b_fc2[84];
    float b_fc3[10];
    float probs[10];

    int i, j, m, n, index;
    FILE *fp;

    /* Load Weights from DDR->LMM */
    fp = fopen("data/weights/w_conv1.txt", "r");
    for(i=0;i<6;i++)
        fscanf(fp, "%f ",  &(w_conv1[i][0][0]));  fclose(fp);

    fp = fopen("data/weights/w_conv2.txt", "r");
    for(i=0;i<16;i++){
        for(j=0;j<6;j++){
            for(m=0;m<5;m++){
                for(n=0;n<5;n++){
                    index = 16*i + 6*j + 5*m + 5*n;
                    fscanf(fp, "%f ",  &(w_conv2[i][j][m][n]));
                }
            }
        }
    }
    fclose(fp);

    fp = fopen("data/weights/w_fc1.txt", "r");
    for(i=0;i<120;i++){
        for(j=0;j<400;j++)
            fscanf(fp, "%f ",  &(w_fc1[i][j]));
    }
    fclose(fp);

    fp = fopen("data/weights/w_fc2.txt", "r");
    for(i=0;i<84;i++){
        for(j=0;j<120;j++)
            fscanf(fp, "%f ",  &(w_fc2[i][j]));
    }
    fclose(fp);

    fp = fopen("data/weights/w_fc3.txt", "r");
    for(i=0;i<10;i++){
        for(j=0;j<84;j++)
            fscanf(fp, "%f ",  &(w_fc3[i][j]));
    }
    fclose(fp);

    fp = fopen("data/weights/b_conv1.txt", "r");
    for(i=0;i<6;i++)
        fscanf(fp, "%f ",  &(b_conv1[i]));  fclose(fp);

    fp = fopen("data/weights/b_conv2.txt", "r");
    for(i=0;i<16;i++)
        fscanf(fp, "%f ",  &(b_conv2[i]));  fclose(fp);

    fp = fopen("data/weights/b_fc1.txt", "r");
    for(i=0;i<120;i++)
        fscanf(fp, "%f ",  &(b_fc1[i]));  fclose(fp);

    fp = fopen("data/weights/b_fc2.txt", "r");
    for(i=0;i<84;i++)
        fscanf(fp, "%f ",  &(b_fc2[i]));  fclose(fp);

    fp = fopen("data/weights/b_fc3.txt", "r");
    for(i=0;i<10;i++)
        fscanf(fp, "%f ",  &(b_fc3[i]));  fclose(fp);

    float *dataset = (float*)malloc(LABEL_LEN*28*28 *sizeof(float));
    int target[LABEL_LEN];

    fp = fopen("mnist-test-target.txt", "r");
    for(i=0;i<LABEL_LEN;i++)
        fscanf(fp, "%d ",  &(target[i]));  fclose(fp);

    fp = fopen("mnist-test-image.txt", "r");
    for(i=0;i<LABEL_LEN*28*28;i++)
        fscanf(fp, "%f ",  &(dataset[i]));  fclose(fp);

    float image[28][28];
    float *datain;
    int acc = 0;
    int mm, nn;

    // Using 2 different quantization parameters for weights and activations

    // Experiment with small scale values based on recommend on the paper https://arxiv.org/abs/2106.08295

    QuantParams weight_params = {0.1f, 8, 0}; 
    QuantParams act_params = {0.5f, 8, 0};  
    for(i=0;i<LABEL_LEN;i++) {
        datain = &dataset[i*28*28];
        for(mm=0;mm<28;mm++)
            for(nn=0;nn<28;nn++)
                image[mm][nn] = *(float*)&datain[28*mm + nn];

        Prediction(   image,
                    w_conv1,
                    w_conv2,
                    w_fc1,
                    w_fc2,
                    w_fc3,
                    b_conv1,
                    b_conv2,
                    b_fc1,
                    b_fc2,
                    b_fc3,
                    probs,
                    weight_params,
                    act_params
                    );

        int index = 0;
        float max = probs[0];
        for (j=1;j<10;j++) {
            if (probs[j] > max) {
                index = j;
                max = probs[j];
            }
        }

       if (index == target[i]) acc++;
       printf("Predicted label: %d\n", index);
       printf("Prediction: %d/%d\n", acc, i+1);
   }
   printf("Accuracy = %f\n", acc*1.0f/LABEL_LEN);

    return 0;
}

void Prediction(float image[28][28], float w_conv1[6][1][1], float w_conv2[16][6][5][5], float w_fc1[120][400], float w_fc2[84][120], float w_fc3[10][84], float b_conv1[6], float b_conv2[16], float b_fc1[120], float b_fc2[84], float b_fc3[10], float probs[10], QuantParams weight_params, QuantParams act_params) {

    int32_t conv1_out[6][28][28] = {0};
    int32_t pool1_out[6][14][14] = {0};

    // Convolution 1
     for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {
                int quantized_weight = quantize(w_conv1[c][0][0], weight_params);
                conv1_out[c][h][w] = (int32_t)(image[h][w] * quantized_weight);
            }
        }
    }

   // Add bias (float) after compute convolution 1
    for(int c=0; c < 6; c++){
        for(int h=0; h<28; h++){
            for(int w=0; w<28; w++){
               conv1_out[c][h][w] += (int32_t)b_conv1[c] ;
            }
         }
    }

    // ReLU activation
    for (int c = 0; c < 6; c++) {
        relu_int(&conv1_out[c][0][0], 28 * 28);
    }


    // Average Pooling 1 
     for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
               pool1_out[c][h][w] = (conv1_out[c][2*h][2*w] + conv1_out[c][2*h+1][2*w] + conv1_out[c][2*h][2*w+1] + conv1_out[c][2*h+1][2*w+1]);
              if(pool1_out[c][h][w] > 0) pool1_out[c][h][w] = pool1_out[c][h][w] >> 2; //right shift 2 bits ~ divide by 4
            }
        }
    }


    int32_t conv2_out[16][14][14] = {0};
    int32_t pool2_out[16][5][5] = {0};


    // Convolution 2
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
                 for (int k = 0; k < 6; k++) {
                   for (int m = 0; m < 5; m++) {
                        for (int n = 0; n < 5; n++) {
                           int quantized_weight = quantize(w_conv2[c][k][m][n], weight_params);
                            conv2_out[c][h][w] += (pool1_out[k][h+m][w+n] * quantized_weight);
                        }
                   }
                }
            }
        }
    }
        // Add bias (float) to the output of convolution 2
        for (int c = 0; c < 16; c++) {
            for (int h = 0; h < 14; h++) {
                for (int w = 0; w < 14; w++) {
                     conv2_out[c][h][w] += (int32_t)b_conv2[c];
                }
            }
        }


   // ReLU activation
    for (int c = 0; c < 16; c++) {
        relu_int(&conv2_out[c][0][0], 14 * 14);
    }

    // Average Pooling 2
     for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 5; h++) {
            for (int w = 0; w < 5; w++) {
                pool2_out[c][h][w] = (conv2_out[c][2*h][2*w] + conv2_out[c][2*h+1][2*w] + conv2_out[c][2*h][2*w+1] + conv2_out[c][2*h+1][2*w+1]);
                if(pool2_out[c][h][w] > 0)  pool2_out[c][h][w] = pool2_out[c][h][w] >> 2; //right shift 2 bits ~ divide by 4

           }
        }
    }

    // Flatten the output from pooling layer
    int32_t flat_out[16 * 5 * 5] = {0};
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 5; h++) {
            for (int w = 0; w < 5; w++) {
                flat_out[c * 5 * 5 + h * 5 + w] = pool2_out[c][h][w];
            }
        }
    }

   // Fully Connected Layer 1
     int32_t fc1_out[120] = {0};
    for (int i = 0; i < 120; i++) {
        for (int j = 0; j < 400; j++) {
            int quantized_weight = quantize(w_fc1[i][j], weight_params);
            fc1_out[i] += (flat_out[j] * quantized_weight);
        }
    }
    // Add bias (float) after compute fully connected layer 1
    for(int i=0; i < 120; i++){
        fc1_out[i] += (int32_t)b_fc1[i];
    }


   // ReLU activation
    relu_int(fc1_out, 120);


   // Fully Connected Layer 2
    int32_t fc2_out[84] = {0};
    for (int i = 0; i < 84; i++) {
        for (int j = 0; j < 120; j++) {
            int quantized_weight = quantize(w_fc2[i][j], weight_params);
            fc2_out[i] += (fc1_out[j] * quantized_weight);
        }
    }
    // Add bias (float) affter compute fully connected layer 2
    for (int i = 0; i < 84; i++) {
            fc2_out[i] += (int32_t)b_fc2[i];
        }
    // ReLU activation
    relu_int(fc2_out, 84);


    // Fully Connected Layer 3
    int32_t fc3_out[10] = {0};
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 84; j++) {
            int quantized_weight = quantize(w_fc3[i][j], weight_params);
            fc3_out[i] += (fc2_out[j] * quantized_weight);
         }
    }
    // Add bias (float) after compute fully connected layer 3
    for (int i = 0; i < 10; i++) {
        fc3_out[i] += (int32_t)b_fc3[i];
    }


    // Softmax activation
    softmax_int(fc3_out, probs, 10, act_params);
}