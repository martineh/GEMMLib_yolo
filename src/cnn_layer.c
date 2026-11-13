#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "gemm_blis.h"
#include "cnn_layer.h"


void update_cnn(cnn_test_t *cnn_test, cnn_model_t *model) {
  for (int l = 0;  l < cnn_test->layers; l++) {
    cnn_test->cnn_model[l].Co = model[l].Co; 
    cnn_test->cnn_model[l].Wo = model[l].Wo; 
    cnn_test->cnn_model[l].Ho = model[l].Ho; 
    cnn_test->cnn_model[l].Wf = model[l].Wf; 
    cnn_test->cnn_model[l].Hf = model[l].Hf; 
    cnn_test->cnn_model[l].Ci = model[l].Ci; 
    cnn_test->cnn_model[l].Hi = model[l].Hi; 
    cnn_test->cnn_model[l].Wi = model[l].Wi; 
  }
}


cnn_test_t *new_cnn_test(int model) {

  cnn_test_t *cnn_test = (cnn_test_t *)malloc(sizeof(cnn_test_t));

  if (model == RESNET50_V15) {
    cnn_test->layers = 20;
    cnn_model_t model[20] = {
    // Co,  Wo, Ho, Wf, Hf, Ci
    {  64, 112,112,  7,  7,   3,  224, 224},  //C1
    {  64,  56, 56,  1,  1,  64,  112, 112},  //C2
    {  64,  56, 56,  3,  3,  64,   56,  56},  //C3
    { 256,  56, 56,  1,  1,  64,   56,  56},  //C4
    {  64,  56, 56,  1,  1, 256,   56,  56},  //C5
    { 128,  56, 56,  1,  1, 256,   56,  56},  //C6
    { 128,  28, 28,  3,  3, 128,   56,  56},  //C7
    { 512,  28, 28,  1,  1, 128,   28,  28},  //C8
    { 512,  28, 28,  1,  1, 256,   28,  28},  //C9
    { 128,  28, 28,  1,  1, 512,   28,  28},  //C10
    { 256,  28, 28,  1,  1, 512,   28,  28},  //C11
    { 256,  14, 14,  3,  3, 256,   28,  28},  //C12
    {1024,  14, 14,  1,  1, 256,   14,  14},  //C13
    {1024,  14, 14,  1,  1, 512,   14,  14},  //C14
    { 256,  14, 14,  1,  1, 1024,  14,  14},  //C15
    { 512,  14, 14,  1,  1, 1024,  14,  14},  //C16
    { 512,   7,  7,  3,  3, 512,   14,  14},  //C17 
    {2048,   7,  7,  1,  1, 512,    7,   7},  //C18
    {2048,   7,  7,  1,  1, 1024,   7,   7},  //C19
    { 512,   7,  7,  1,  1, 2048,   7,   7}   //C20
    };

    update_cnn(cnn_test, model);

  } else if (model == VGG16) {
    cnn_test->layers = 9;
    cnn_model_t model[10] = {
    // Co,  Wo, Ho, Wf, Hf, Ci
    {  64, 224,224,  3,  3,   3, 224, 224},  //C1
    {  64, 224,224,  3,  3,  64, 224, 224},  //C2
    { 128, 112,112,  3,  3,  64, 112, 112},  //C3
    { 128, 112,112,  3,  3, 128, 112, 112},  //C4
    { 256,  56, 56,  3,  3, 128,  56,  56},  //C5
    { 256,  56, 56,  3,  3, 256,  56,  56},  //C6
    { 512,  28, 28,  3,  3, 256,  28,  28},  //C7
    { 512,  28, 28,  3,  3, 512,  28,  28},  //C8
    { 512,  14, 14,  3,  3, 512,  14,  14}   //C9
    };
    
    update_cnn(cnn_test, model);

  } else {
    printf("ERROR: Model unsuported\n"); exit(-1);
  }


  return cnn_test;

}

void free_cnn_test(cnn_test_t *cnn_test) {
  free(cnn_test);
}



//Quantization
void sym_qt_int32_to_int8(float *scale, int32_t *DQ, int8_t *Q, size_t m, size_t n) {
  size_t nbits = sizeof(int8_t) * 8;

  int32_t max_abs = abs(DQ[0]); 
  #pragma omp parallel for reduction(max:max_abs)
  for (int i = 0; i < m * n; i++) 
    if (abs(DQ[i]) > max_abs) max_abs = abs(DQ[i]);
  float s = max_abs / pow(2, nbits - 1);
  
  int32_t betaq  = pow(2,  nbits - 1) - 1;
  int32_t alphaq = pow(-2, nbits - 1);
 
  #pragma omp parallel for 
  for (int i = 0; i < m * n; i++) 
    clip(alphaq, betaq, (int32_t)round(DQ[i] / s), Q[i]);

  *scale = s;

}

void sym_qt_float_to_int8(float *scale, float *DQ, int8_t *Q, size_t m, size_t n) {
  size_t nbits = sizeof(int8_t) * 8;

  float max_abs = abs(DQ[0]); 
  #pragma omp parallel for reduction(max:max_abs)
  for (int i = 0; i < m * n; i++) 
    if (abs(DQ[i]) > max_abs) max_abs = abs(DQ[i]);

  float s = max_abs / pow(2, nbits - 1);
  
  float betaq  = pow(2,  nbits - 1) - 1;
  float alphaq = pow(-2, nbits - 1);

  #pragma omp parallel for 
  for (int i = 0; i < m * n; i++) 
    clip(alphaq, betaq, (float)round(DQ[i] / s), Q[i]);

  *scale = s;

}

void sym_qt_float_to_int4(float *scale, float *DQ, int8_t *Q, size_t m, size_t n) {
  size_t nbits = 4 * 8;

  float max_abs = abs(DQ[0]); 
  #pragma omp parallel for reduction(max:max_abs)
  for (int i = 0; i < m * n; i++) 
    if (abs(DQ[i]) > max_abs) max_abs = abs(DQ[i]);

  float s = max_abs / pow(2, nbits - 1);
  
  float betaq  = pow(2,  nbits - 1) - 1;
  float alphaq = pow(-2, nbits - 1);

  #pragma omp parallel for 
  for (int i = 0; i < m * n; i++) 
    clip(alphaq, betaq, (float)round(DQ[i] / s), Q[i]);

  *scale = s;

}





//Dequantization
void sym_dqt_int32_to_float(float scale, int32_t * Q, float *DQ, size_t m, size_t n) {
  #pragma omp parallel for 
  for (int i = 0; i < m * n; i++) DQ[i] = (float)Q[i] * scale;
}

void sym_dqt_int8_to_float(float scale, int8_t *Q, float *DQ, size_t m, size_t n) {
  #pragma omp parallel for 
  for (int i = 0; i < m * n; i++) DQ[i] = (float)Q[i] * scale;
}

void sym_dqt_int4_to_float(float scale, int8_t *Q, float *DQ, size_t m, size_t n) {
  #pragma omp parallel for 
  for (int i = 0; i < m * n; i++) DQ[i] = (float)Q[i] * scale;
}





//Transformers Layers. Not parallelized! Output parallelization.
void softlogmax(float *C, float *A, size_t m, size_t n) {
  size_t size = m * n;
    
  float max_val = A[0];
  for (size_t i = 0; i < size; i++)
    if (A[i] > max_val) max_val = A[i];
  
  float log_sum_exp = 0.0;
  for (size_t i = 0; i < size; i++) {
    A[i] = exp(A[i] - max_val);
    log_sum_exp += A[i];
  }

  log_sum_exp = log(log_sum_exp);

  for (size_t i = 0; i < size; i++)
    C[i] = A[i] - log_sum_exp;

}


void add_layernorm(int order, float *C, float *A, float *B, size_t batch, size_t m, size_t n, float *running_mean, float *inv_std) {
  
  size_t i, j;
  size_t bsize = n / batch;
  
  if (order != COLUMN_MAJOR) { printf("ROW-Major not implemented\n"); exit(-1); }

  #pragma omp parallel for private(i) 
  for (j=0; j < n; j++) {
    size_t b = j / bsize;
    for (i=0; i < m; i++) {
      C[(j * m) + i] = ((A[(j * m) + i] + B[(j * m) + i]) - running_mean[b]) * inv_std[b];
    }
  }


}

void gelu(float *C, float *A, size_t m, size_t n) {
  //https://github.com/karfaoui/gelu/blob/master/src/Activation.cpp#L61
  //https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
  //We detect hard time variations using the funcion tanh depending with
  //the vector values. 
  //We need a time stability, for this reason we implement a new gelu aproximation.
  //const float SQRTtmp = 2.0 / PI;
  //#pragma omp parallel for 
  //for (i=0; i < m * n; i++) C[i] = 0.5 * A[i] * (1 + tanh(sqrt(2.0 / PI) * (A[i] + Al * (A[i]*A[i]*A[i])) ));
  
  //Gelu aproximation without tanh call.
  #pragma omp parallel for
  for (int i=0; i < m * n; i++) C[i] = 0.5 * A[i] * (1 + (A[i]/(1 + fabs(A[i]))));

}

