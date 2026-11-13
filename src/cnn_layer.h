#ifndef __CNN_LAYER__
#define __CNN_LAYER__

#define PI  3.14159265358979323846
#define Al  0.044715

#define abs(a)  ( (a) > 0.0 ? (a) : -(a) )

#define clip(min, max, val, target) {\
  if      (val < min) target=min;\
  else if (val > max) target=max;\
  else                target=val;\
}

//CNN Models
#define LOWERING_NHWC 0
#define LOWERING_NCHW 1
#define GEMM          2
#define RESNET50_V15  0
#define VGG16         1

#define MAX_LAYERS    50

#define NHWC          0
#define NCHW          1

typedef struct cnn_model {
  int Co;
  int Wo;
  int Ho;
  int Wf;
  int Hf;
  int Ci;
  int Wi;
  int Hi;
} cnn_model_t;

typedef struct cnn_test {
  cnn_model_t cnn_model[MAX_LAYERS];
  int layers;
} cnn_test_t;


cnn_test_t *new_cnn_test(int model);
void free_cnn_test(cnn_test_t *cnn_test);

void sym_qt_int32_to_int8(float *scale, int32_t *DQ, int8_t *Q, size_t m, size_t n);
void sym_qt_float_to_int8(float *scale, float *DQ, int8_t *Q, size_t m, size_t n);
void sym_qt_float_to_int4(float *scale, float *DQ, int8_t *Q, size_t m, size_t n);
void sym_dqt_int32_to_float(float scale, int32_t * Q, float *DQ, size_t m, size_t n);
void sym_dqt_int8_to_float(float scale, int8_t *Q, float *DQ, size_t m, size_t n);
void sym_dqt_int4_to_float(float scale, int8_t *Q, float *DQ, size_t m, size_t n);

void softlogmax(float *C, float *A, size_t m, size_t n);
void add_layernorm(int order, float *C, float *A, float *B, size_t batch, size_t m, size_t n, float *running_mean, float *inv_std);
void gelu(float *C, float *A, size_t m, size_t n);

#endif

