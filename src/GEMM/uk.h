#ifndef UK_H
#define UK_H

#include "gemm_blis.h"
#include <riscv_vector.h>

void ukernel_16x8_fp32 (int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_16x4_fp32 (int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_16x1_fp32 (int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_64x1_fp32 (int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);

void ukernel_8x8_fp32  (int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_32x8_fp32 (int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_8x16_fp32 (int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_16x16_fp32(int mr, int nr, int kc, const float  *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda);

void ukernel_16x8_fp16 (int mr, int nr, int kc, const fp16_t  *Ar, const fp16_t *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_16x4_fp16 (int mr, int nr, int kc, const fp16_t  *Ar, const fp16_t *Br, float *Cr, float beta, int ldA, int ldB, int Clda);

void ukernel_32x8_fp16 (int mr, int nr, int kc, const fp16_t  *Ar, const fp16_t *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void ukernel_32x4_fp16 (int mr, int nr, int kc, const fp16_t  *Ar, const fp16_t *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
#endif
