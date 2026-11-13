
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include "uk.h"
#include "modelLevel/model_level.h"

// GEMM TYPES
#define ROW_MAJOR    101 
#define COLUMN_MAJOR 102 

#define TRANSPOSE    112
#define NO_TRANSPOSE 111

#define B3A2C0 1
#define A3B2C0 2

#define LOOP_JR 4
#define LOOP_IR 5

#define YOLO_MC 256
#define YOLO_NC 512
#define YOLO_KC 256

#ifdef NQ_FP32
  #define YOLO_MR 16
  #define YOLO_NR 8
#else
  #define YOLO_MR 32
  #define YOLO_NR 8
#endif


typedef _Float16 fp16_t;

//We need define this function types here to select the correct micro-kernel header.
typedef void (*uk_intrinsic_fp32)      (int, int, int, const float *, const float *, float *, float, int, int, int);
typedef void (*uk_intrinsic_fp16)      (int, int, int, const fp16_t *, const fp16_t *, float *, float, int, int, int);

//Data types
//+================================================================================================+
//|                               QUANTIZATION CONFIGURATIONS                                      |
//+================+========================================+======================================+
//|  NQ_FP32       | Input:FP32  + Pack:FP32  + Micro:FP32  | No quantization with FP32.           |
//|  NQ_FP16       | Input:FP16  + Pack:FP16  + Micro:FP16  | No quantization with FP16.           |
//|  FQ_INT8_FP32  | Input:INT8  + Pack:FP32  + Micro:FP32  | Fake quantization. Casting to FP32.  |
//|  Q_INT8_INT32  | Input:INT8  + Pack:INT8  + Micro:INT32 | Quantization.                        |
//+----------------+----------------------------------------+--------------------------------------+
//|                | Input:INT8  + Pack:INT32 + Micro:INT32 | Fake quantization. Casting to INT32. |
//|  FQ_INT8_INT32 | WARNING: RVV0.7 without support for    | WARNING: This configuration is only  |
//|                | quantization                           | available for C910 and C906.         |
//+================+========================================+======================================+

 
#if defined(NQ_FP32) | defined(NQ_FP16)
  #define IN_TYPE        float 
  #define OUT_TYPE       float 

  #if NQ_FP16
    #define PACK_TYPE      fp16_t
    #define uk_intrinsic   uk_intrinsic_fp16
  #else
    #define PACK_TYPE      float 
    #define uk_intrinsic   uk_intrinsic_fp32
  #endif

#elif FQ_INT8_FP32
  #define IN_TYPE        int8_t
  #define PACK_TYPE      float
  #define OUT_TYPE       float
  #define uk_intrinsic   uk_intrinsic_fp32
#elif FQ_INT8_INT32
  #define IN_TYPE        int8_t
  #define PACK_TYPE      int32_t
  #define OUT_TYPE       int32_t
  #define uk_intrinsic   uk_intrinsic_int32
#else
  #define IN_TYPE        int8_t
  #define PACK_TYPE      int8_t
  #define OUT_TYPE       int32_t
  #define uk_intrinsic   uk_intrinsic_int8_int32
#endif

#define BCH_IN_TYPE   float
#define BCH_PACK_TYPE float
#define BCH_OUT_TYPE  float
  
#if defined(NQ_FP16) || defined(FQ_INT8_FP32) || defined(Q_INT8_INT32) || defined(FQ_INT8_INT32)
  #define QUANTIZE_ENABLE
#else
  #define QUANTIZE_DISABLE
#endif


//Prototypes
void prepack_A( char orderA, size_t m, size_t k, IN_TYPE *A, size_t ldA, PACK_TYPE *Ac,
                size_t MC, size_t KC, int MR);


double gemm_validation(char order, char transA, char transB, IN_TYPE *A, IN_TYPE *B, 
		       int m, int n, int k, OUT_TYPE alpha, OUT_TYPE beta, int ldA, int ldB, 
		       int ldC, OUT_TYPE *C, OUT_TYPE *Cg);

void gemm_rvv(char *transa, char *transb, int *m, int *n, int *k, OUT_TYPE *alpha, 
	      const IN_TYPE *a, int *lda, const IN_TYPE *b, int *ldb, OUT_TYPE *beta, 
	      OUT_TYPE *c, int *ldc);

void silu_kernel_rvv(const float* input, float* output, size_t n);

void im2col_pack_rvv(const float* input_data, float* finput_data,
                     int64_t kH, int64_t kW, int64_t dH, int64_t dW, int64_t padH, int64_t padW,
                     int64_t n_input_plane, int64_t input_height, int64_t input_width,
                     int64_t output_height, int64_t output_width);

