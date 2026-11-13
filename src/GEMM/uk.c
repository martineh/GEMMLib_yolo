#include "uk.h"

#include <stdio.h>

#define Crref(i,j)  Cr[j*Clda+i]
#define min(a,b) a >= b ? a : b
#define vinit(vreg, value, vlength)        vreg = __riscv_vfmv_v_f_f32m1(value, vlength)
#define vloadC(vreg, mem,  vlength)        vreg = __riscv_vle32_v_f32m1 (mem, vlength)
#define vstoreC(mem, vreg, vlength)               __riscv_vse32_v_f32m1 (mem, vreg, vlength)
#define vload(vreg, vtmp, mem, vlength)    vreg = __riscv_vle32_v_f32m1 (mem, vlength); 
#define vdup(vreg, mem, vlength)           vreg = __riscv_vfmv_v_f_f32m1(mem, vlength)
#define vupdate(vreg1, vreg2, value, vlength) vreg1=__riscv_vfmacc_vf_f32m1(vreg1, value, vreg2, vlength)

#include <riscv_vector.h>

#include <riscv_vector.h>

void ukernel_16x16_fp32(
    int mr, int nr, int kc,
    const float *Ar, const float *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e32m2(16);

    vfloat32m2_t A0;

    vfloat32m2_t C00, C01, C02, C03;
    vfloat32m2_t C04, C05, C06, C07;
    vfloat32m2_t C08, C09, C10, C11;
    vfloat32m2_t C12, C13, C14, C15;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C01 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C02 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C03 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C04 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C05 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C06 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C07 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C08 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C09 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C10 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C11 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C12 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C13 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C14 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C15 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m2(&Crref(0,  0), vl);
        C01 = __riscv_vle32_v_f32m2(&Crref(0,  1), vl);
        C02 = __riscv_vle32_v_f32m2(&Crref(0,  2), vl);
        C03 = __riscv_vle32_v_f32m2(&Crref(0,  3), vl);
        C04 = __riscv_vle32_v_f32m2(&Crref(0,  4), vl);
        C05 = __riscv_vle32_v_f32m2(&Crref(0,  5), vl);
        C06 = __riscv_vle32_v_f32m2(&Crref(0,  6), vl);
        C07 = __riscv_vle32_v_f32m2(&Crref(0,  7), vl);
        C08 = __riscv_vle32_v_f32m2(&Crref(0,  8), vl);
        C09 = __riscv_vle32_v_f32m2(&Crref(0,  9), vl);
        C10 = __riscv_vle32_v_f32m2(&Crref(0, 10), vl);
        C11 = __riscv_vle32_v_f32m2(&Crref(0, 11), vl);
        C12 = __riscv_vle32_v_f32m2(&Crref(0, 12), vl);
        C13 = __riscv_vle32_v_f32m2(&Crref(0, 13), vl);
        C14 = __riscv_vle32_v_f32m2(&Crref(0, 14), vl);
        C15 = __riscv_vle32_v_f32m2(&Crref(0, 15), vl);
    }

    for (pr = 0; pr < kc; ++pr) {
        A0 = __riscv_vle32_v_f32m2(&Ar[bA], vl);

        C00 = __riscv_vfmacc_vf_f32m2(C00, Br[bB +  0], A0, vl);
        C01 = __riscv_vfmacc_vf_f32m2(C01, Br[bB +  1], A0, vl);
        C02 = __riscv_vfmacc_vf_f32m2(C02, Br[bB +  2], A0, vl);
        C03 = __riscv_vfmacc_vf_f32m2(C03, Br[bB +  3], A0, vl);
        C04 = __riscv_vfmacc_vf_f32m2(C04, Br[bB +  4], A0, vl);
        C05 = __riscv_vfmacc_vf_f32m2(C05, Br[bB +  5], A0, vl);
        C06 = __riscv_vfmacc_vf_f32m2(C06, Br[bB +  6], A0, vl);
        C07 = __riscv_vfmacc_vf_f32m2(C07, Br[bB +  7], A0, vl);
        C08 = __riscv_vfmacc_vf_f32m2(C08, Br[bB +  8], A0, vl);
        C09 = __riscv_vfmacc_vf_f32m2(C09, Br[bB +  9], A0, vl);
        C10 = __riscv_vfmacc_vf_f32m2(C10, Br[bB + 10], A0, vl);
        C11 = __riscv_vfmacc_vf_f32m2(C11, Br[bB + 11], A0, vl);
        C12 = __riscv_vfmacc_vf_f32m2(C12, Br[bB + 12], A0, vl);
        C13 = __riscv_vfmacc_vf_f32m2(C13, Br[bB + 13], A0, vl);
        C14 = __riscv_vfmacc_vf_f32m2(C14, Br[bB + 14], A0, vl);
        C15 = __riscv_vfmacc_vf_f32m2(C15, Br[bB + 15], A0, vl);

        bA += 16;
        bB += 16;
    }

    __riscv_vse32_v_f32m2(&Crref(0,  0), C00, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  1), C01, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  2), C02, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  3), C03, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  4), C04, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  5), C05, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  6), C06, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  7), C07, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  8), C08, vl);
    __riscv_vse32_v_f32m2(&Crref(0,  9), C09, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 10), C10, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 11), C11, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 12), C12, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 13), C13, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 14), C14, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 15), C15, vl);
}

void ukernel_16x8_fp32(
    int mr, int nr, int kc,
    const float *Ar, const float *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e32m2(16);

    vfloat32m2_t A0;
    vfloat32m2_t C00, C01, C02, C03;
    vfloat32m2_t C04, C05, C06, C07;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C01 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C02 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C03 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C04 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C05 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C06 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C07 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m2(&Crref(0, 0), vl);
        C01 = __riscv_vle32_v_f32m2(&Crref(0, 1), vl);
        C02 = __riscv_vle32_v_f32m2(&Crref(0, 2), vl);
        C03 = __riscv_vle32_v_f32m2(&Crref(0, 3), vl);
        C04 = __riscv_vle32_v_f32m2(&Crref(0, 4), vl);
        C05 = __riscv_vle32_v_f32m2(&Crref(0, 5), vl);
        C06 = __riscv_vle32_v_f32m2(&Crref(0, 6), vl);
        C07 = __riscv_vle32_v_f32m2(&Crref(0, 7), vl);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle32_v_f32m2(&Ar[bA], vl);

        C00 = __riscv_vfmacc_vf_f32m2(C00, Br[bB + 0], A0, vl);
        C01 = __riscv_vfmacc_vf_f32m2(C01, Br[bB + 1], A0, vl);
        C02 = __riscv_vfmacc_vf_f32m2(C02, Br[bB + 2], A0, vl);
        C03 = __riscv_vfmacc_vf_f32m2(C03, Br[bB + 3], A0, vl);
        C04 = __riscv_vfmacc_vf_f32m2(C04, Br[bB + 4], A0, vl);
        C05 = __riscv_vfmacc_vf_f32m2(C05, Br[bB + 5], A0, vl);
        C06 = __riscv_vfmacc_vf_f32m2(C06, Br[bB + 6], A0, vl);
        C07 = __riscv_vfmacc_vf_f32m2(C07, Br[bB + 7], A0, vl);

        bA += 16;
        bB += 8;
    }

    __riscv_vse32_v_f32m2(&Crref(0, 0), C00, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 1), C01, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 2), C02, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 3), C03, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 4), C04, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 5), C05, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 6), C06, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 7), C07, vl);
}

void ukernel_16x4_fp32(
    int mr, int nr, int kc,
    const float *Ar, const float *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e32m2(16);

    vfloat32m2_t A0;
    vfloat32m2_t C00, C01, C02, C03;
    vfloat32m2_t C04, C05, C06, C07;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C01 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C02 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C03 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m2(&Crref(0, 0), vl);
        C01 = __riscv_vle32_v_f32m2(&Crref(0, 1), vl);
        C02 = __riscv_vle32_v_f32m2(&Crref(0, 2), vl);
        C03 = __riscv_vle32_v_f32m2(&Crref(0, 3), vl);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle32_v_f32m2(&Ar[bA], vl);

        C00 = __riscv_vfmacc_vf_f32m2(C00, Br[bB + 0], A0, vl);
        C01 = __riscv_vfmacc_vf_f32m2(C01, Br[bB + 1], A0, vl);
        C02 = __riscv_vfmacc_vf_f32m2(C02, Br[bB + 2], A0, vl);
        C03 = __riscv_vfmacc_vf_f32m2(C03, Br[bB + 3], A0, vl);

        bA += 16;
        bB += 4;
    }

    __riscv_vse32_v_f32m2(&Crref(0, 0), C00, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 1), C01, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 2), C02, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 3), C03, vl);
}

void ukernel_16x1_fp32(
    int mr, int nr, int kc,
    const float *Ar, const float *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e32m2(16);

    vfloat32m2_t A0;
    vfloat32m2_t C00, C01, C02, C03;
    vfloat32m2_t C04, C05, C06, C07;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m2(&Crref(0, 0), vl);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle32_v_f32m2(&Ar[bA], vl);

        C00 = __riscv_vfmacc_vf_f32m2(C00, Br[bB + 0], A0, vl);

        bA += 16;
        bB += 1;
    }

    __riscv_vse32_v_f32m2(&Crref(0, 0), C00, vl);
}

void ukernel_64x1_fp32(
    int mr, int nr, int kc,
    const float *Ar, const float *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e32m8(64);

    vfloat32m8_t A0;
    vfloat32m8_t C00, C01, C02, C03;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m8(&Crref(0, 0), vl);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle32_v_f32m8(&Ar[bA], vl);

        C00 = __riscv_vfmacc_vf_f32m8(C00, Br[bB + 0], A0, vl);

        bA += 64;
        bB += 1;
    }

    __riscv_vse32_v_f32m8(&Crref(0, 0), C00, vl);
}

void ukernel_32x8_fp32(
    int mr, int nr, int kc,
    const float *Ar, const float *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e32m4(32);

    vfloat32m4_t A0;
    vfloat32m4_t C00, C01, C02, C03;
    vfloat32m4_t C04, C05, C06, C07;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        C01 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        C02 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        C03 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        C04 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        C05 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        C06 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        C07 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m4(&Crref(0, 0), vl);
        C01 = __riscv_vle32_v_f32m4(&Crref(0, 1), vl);
        C02 = __riscv_vle32_v_f32m4(&Crref(0, 2), vl);
        C03 = __riscv_vle32_v_f32m4(&Crref(0, 3), vl);
        C04 = __riscv_vle32_v_f32m4(&Crref(0, 4), vl);
        C05 = __riscv_vle32_v_f32m4(&Crref(0, 5), vl);
        C06 = __riscv_vle32_v_f32m4(&Crref(0, 6), vl);
        C07 = __riscv_vle32_v_f32m4(&Crref(0, 7), vl);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle32_v_f32m4(&Ar[bA], vl);

        C00 = __riscv_vfmacc_vf_f32m4(C00, Br[bB + 0], A0, vl);
        C01 = __riscv_vfmacc_vf_f32m4(C01, Br[bB + 1], A0, vl);
        C02 = __riscv_vfmacc_vf_f32m4(C02, Br[bB + 2], A0, vl);
        C03 = __riscv_vfmacc_vf_f32m4(C03, Br[bB + 3], A0, vl);
        C04 = __riscv_vfmacc_vf_f32m4(C04, Br[bB + 4], A0, vl);
        C05 = __riscv_vfmacc_vf_f32m4(C05, Br[bB + 5], A0, vl);
        C06 = __riscv_vfmacc_vf_f32m4(C06, Br[bB + 6], A0, vl);
        C07 = __riscv_vfmacc_vf_f32m4(C07, Br[bB + 7], A0, vl);

        bA += 32;
        bB += 8;
    }

    __riscv_vse32_v_f32m4(&Crref(0, 0), C00, vl);
    __riscv_vse32_v_f32m4(&Crref(0, 1), C01, vl);
    __riscv_vse32_v_f32m4(&Crref(0, 2), C02, vl);
    __riscv_vse32_v_f32m4(&Crref(0, 3), C03, vl);
    __riscv_vse32_v_f32m4(&Crref(0, 4), C04, vl);
    __riscv_vse32_v_f32m4(&Crref(0, 5), C05, vl);
    __riscv_vse32_v_f32m4(&Crref(0, 6), C06, vl);
    __riscv_vse32_v_f32m4(&Crref(0, 7), C07, vl);
}


void ukernel_8x8_fp32(int mr, int nr, int kc,
                      const float *Ar, const float *Br, float *Cr,
                      float beta, int ldA, int ldB, int Clda) {
    int pr, bA = 0, bB = 0;
    size_t vlength = 8;

    vfloat32m1_t vtmp;
    vfloat32m1_t A0, B0, B1, B2, B3, B4, B5, B6, B7;
    vfloat32m1_t C00, C01, C02, C03, C04, C05, C06, C07;

    if (beta == 0) {
        C00 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C01 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C02 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C03 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C04 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C05 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C06 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C07 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
    } else {
        C00 = __riscv_vle32_v_f32m1(&Crref(0, 0), vlength);
        C01 = __riscv_vle32_v_f32m1(&Crref(0, 1), vlength);
        C02 = __riscv_vle32_v_f32m1(&Crref(0, 2), vlength);
        C03 = __riscv_vle32_v_f32m1(&Crref(0, 3), vlength);
        C04 = __riscv_vle32_v_f32m1(&Crref(0, 4), vlength);
        C05 = __riscv_vle32_v_f32m1(&Crref(0, 5), vlength);
        C06 = __riscv_vle32_v_f32m1(&Crref(0, 6), vlength);
        C07 = __riscv_vle32_v_f32m1(&Crref(0, 7), vlength);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle32_v_f32m1(&Ar[bA + 0], vlength);

        C00 = __riscv_vfmacc_vf_f32m1(C00, Br[bB + 0], A0, vlength);
        C01 = __riscv_vfmacc_vf_f32m1(C01, Br[bB + 1], A0, vlength);
        C02 = __riscv_vfmacc_vf_f32m1(C02, Br[bB + 2], A0, vlength);
        C03 = __riscv_vfmacc_vf_f32m1(C03, Br[bB + 3], A0, vlength);
        C04 = __riscv_vfmacc_vf_f32m1(C04, Br[bB + 4], A0, vlength);
        C05 = __riscv_vfmacc_vf_f32m1(C05, Br[bB + 5], A0, vlength);
        C06 = __riscv_vfmacc_vf_f32m1(C06, Br[bB + 6], A0, vlength);
        C07 = __riscv_vfmacc_vf_f32m1(C07, Br[bB + 7], A0, vlength);

        bA += 8;
        bB += 8;
    }

    __riscv_vse32_v_f32m1(&Crref(0, 0), C00, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 1), C01, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 2), C02, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 3), C03, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 4), C04, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 5), C05, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 6), C06, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 7), C07, vlength);
}

void ukernel_8x16_fp32(int mr, int nr, int kc,
                      const float *Ar, const float *Br, float *Cr,
                      float beta, int ldA, int ldB, int Clda) {
    int pr, bA = 0, bB = 0;
    size_t vlength = 8;

    vfloat32m1_t vtmp;
    vfloat32m1_t A0;

    vfloat32m1_t C00, C01, C02, C03, C04, C05, C06, C07;
    vfloat32m1_t C08, C09, C10, C11, C12, C13, C14, C15;

    if (beta == 0) {
        C00 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C01 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C02 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C03 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C04 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C05 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C06 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C07 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C08 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C09 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C10 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C11 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C12 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C13 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C14 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
        C15 = __riscv_vfmv_v_f_f32m1(0.0f, vlength);
    } else {
        C00 = __riscv_vle32_v_f32m1(&Crref(0, 0), vlength);
        C01 = __riscv_vle32_v_f32m1(&Crref(0, 1), vlength);
        C02 = __riscv_vle32_v_f32m1(&Crref(0, 2), vlength);
        C03 = __riscv_vle32_v_f32m1(&Crref(0, 3), vlength);
        C04 = __riscv_vle32_v_f32m1(&Crref(0, 4), vlength);
        C05 = __riscv_vle32_v_f32m1(&Crref(0, 5), vlength);
        C06 = __riscv_vle32_v_f32m1(&Crref(0, 6), vlength);
        C07 = __riscv_vle32_v_f32m1(&Crref(0, 7), vlength);
        C08 = __riscv_vle32_v_f32m1(&Crref(0, 8), vlength);
        C09 = __riscv_vle32_v_f32m1(&Crref(0, 9), vlength);
        C10 = __riscv_vle32_v_f32m1(&Crref(0, 10), vlength);
        C11 = __riscv_vle32_v_f32m1(&Crref(0, 11), vlength);
        C12 = __riscv_vle32_v_f32m1(&Crref(0, 12), vlength);
        C13 = __riscv_vle32_v_f32m1(&Crref(0, 13), vlength);
        C14 = __riscv_vle32_v_f32m1(&Crref(0, 14), vlength);
        C15 = __riscv_vle32_v_f32m1(&Crref(0, 15), vlength);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle32_v_f32m1(&Ar[bA + 0], vlength);

        C00 = __riscv_vfmacc_vf_f32m1(C00, Br[bB + 0], A0, vlength);
        C01 = __riscv_vfmacc_vf_f32m1(C01, Br[bB + 1], A0, vlength);
        C02 = __riscv_vfmacc_vf_f32m1(C02, Br[bB + 2], A0, vlength);
        C03 = __riscv_vfmacc_vf_f32m1(C03, Br[bB + 3], A0, vlength);
        C04 = __riscv_vfmacc_vf_f32m1(C04, Br[bB + 4], A0, vlength);
        C05 = __riscv_vfmacc_vf_f32m1(C05, Br[bB + 5], A0, vlength);
        C06 = __riscv_vfmacc_vf_f32m1(C06, Br[bB + 6], A0, vlength);
        C07 = __riscv_vfmacc_vf_f32m1(C07, Br[bB + 7], A0, vlength);

        C08 = __riscv_vfmacc_vf_f32m1(C08, Br[bB + 8],  A0, vlength);
        C09 = __riscv_vfmacc_vf_f32m1(C09, Br[bB + 9],  A0, vlength);
        C10 = __riscv_vfmacc_vf_f32m1(C10, Br[bB + 10], A0, vlength);
        C11 = __riscv_vfmacc_vf_f32m1(C11, Br[bB + 11], A0, vlength);
        C12 = __riscv_vfmacc_vf_f32m1(C12, Br[bB + 12], A0, vlength);
        C13 = __riscv_vfmacc_vf_f32m1(C13, Br[bB + 13], A0, vlength);
        C14 = __riscv_vfmacc_vf_f32m1(C14, Br[bB + 14], A0, vlength);
        C15 = __riscv_vfmacc_vf_f32m1(C15, Br[bB + 15], A0, vlength);

        bA += 8;
        bB += 16;
    }

    __riscv_vse32_v_f32m1(&Crref(0, 0), C00, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 1), C01, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 2), C02, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 3), C03, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 4), C04, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 5), C05, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 6), C06, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 7), C07, vlength);
    
    __riscv_vse32_v_f32m1(&Crref(0, 8),  C08, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 9),  C09, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 10), C10, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 11), C11, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 12), C12, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 13), C13, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 14), C14, vlength);
    __riscv_vse32_v_f32m1(&Crref(0, 15), C15, vlength);
}



//------------------------------------
//Micro-kernel for fp16 data type
//------------------------------------

void ukernel_16x8_fp16(
    int mr, int nr, int kc,
    const fp16_t *Ar, const fp16_t *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e16m1(16);      /* 16 con VLEN=256: f16m1 y f32m2 */

    vfloat16m1_t A0;
    vfloat32m2_t C00, C01, C02, C03;
    vfloat32m2_t C04, C05, C06, C07;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C01 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C02 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C03 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C04 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C05 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C06 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C07 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m2(&Crref(0, 0), vl);
        C01 = __riscv_vle32_v_f32m2(&Crref(0, 1), vl);
        C02 = __riscv_vle32_v_f32m2(&Crref(0, 2), vl);
        C03 = __riscv_vle32_v_f32m2(&Crref(0, 3), vl);
        C04 = __riscv_vle32_v_f32m2(&Crref(0, 4), vl);
        C05 = __riscv_vle32_v_f32m2(&Crref(0, 5), vl);
        C06 = __riscv_vle32_v_f32m2(&Crref(0, 6), vl);
        C07 = __riscv_vle32_v_f32m2(&Crref(0, 7), vl);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle16_v_f16m1((const _Float16 *)&Ar[bA], vl);

        C00 = __riscv_vfwmacc_vf_f32m2(C00, Br[bB + 0], A0, vl);
        C01 = __riscv_vfwmacc_vf_f32m2(C01, Br[bB + 1], A0, vl);
        C02 = __riscv_vfwmacc_vf_f32m2(C02, Br[bB + 2], A0, vl);
        C03 = __riscv_vfwmacc_vf_f32m2(C03, Br[bB + 3], A0, vl);
        C04 = __riscv_vfwmacc_vf_f32m2(C04, Br[bB + 4], A0, vl);
        C05 = __riscv_vfwmacc_vf_f32m2(C05, Br[bB + 5], A0, vl);
        C06 = __riscv_vfwmacc_vf_f32m2(C06, Br[bB + 6], A0, vl);
        C07 = __riscv_vfwmacc_vf_f32m2(C07, Br[bB + 7], A0, vl);

        bA += 16;
        bB += 8;
    }

    __riscv_vse32_v_f32m2(&Crref(0, 0), C00, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 1), C01, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 2), C02, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 3), C03, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 4), C04, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 5), C05, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 6), C06, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 7), C07, vl);
}

void ukernel_16x4_fp16(
    int mr, int nr, int kc,
    const fp16_t *Ar, const fp16_t *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e16m1(16);

    vfloat16m1_t A0;
    vfloat32m2_t C00, C01, C02, C03;

    if (beta == 0.0f) {
        C00 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C01 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C02 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        C03 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    } else {
        C00 = __riscv_vle32_v_f32m2(&Crref(0, 0), vl);
        C01 = __riscv_vle32_v_f32m2(&Crref(0, 1), vl);
        C02 = __riscv_vle32_v_f32m2(&Crref(0, 2), vl);
        C03 = __riscv_vle32_v_f32m2(&Crref(0, 3), vl);
    }

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle16_v_f16m1((const _Float16 *)&Ar[bA], vl);

        C00 = __riscv_vfwmacc_vf_f32m2(C00, Br[bB + 0], A0, vl);
        C01 = __riscv_vfwmacc_vf_f32m2(C01, Br[bB + 1], A0, vl);
        C02 = __riscv_vfwmacc_vf_f32m2(C02, Br[bB + 2], A0, vl);
        C03 = __riscv_vfwmacc_vf_f32m2(C03, Br[bB + 3], A0, vl);

        bA += 16;
        bB += 4;
    }

    __riscv_vse32_v_f32m2(&Crref(0, 0), C00, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 1), C01, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 2), C02, vl);
    __riscv_vse32_v_f32m2(&Crref(0, 3), C03, vl);
}



void ukernel_32x8_fp16(
    int mr, int nr, int kc,
    const fp16_t *Ar, const fp16_t *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e16m2(32);
    const fp16_t zero = (fp16_t)0;

    vfloat16m2_t A0;
    vfloat16m2_t C0, C1, C2, C3, C4, C5, C6, C7;

    C0 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C1 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C2 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C3 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C4 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C5 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C6 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C7 = __riscv_vfmv_v_f_f16m2(zero, vl);

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle16_v_f16m2((const _Float16 *)&Ar[bA], vl);

        C0 = __riscv_vfmacc_vf_f16m2(C0, Br[bB + 0], A0, vl);
        C1 = __riscv_vfmacc_vf_f16m2(C1, Br[bB + 1], A0, vl);
        C2 = __riscv_vfmacc_vf_f16m2(C2, Br[bB + 2], A0, vl);
        C3 = __riscv_vfmacc_vf_f16m2(C3, Br[bB + 3], A0, vl);
        C4 = __riscv_vfmacc_vf_f16m2(C4, Br[bB + 4], A0, vl);
        C5 = __riscv_vfmacc_vf_f16m2(C5, Br[bB + 5], A0, vl);
        C6 = __riscv_vfmacc_vf_f16m2(C6, Br[bB + 6], A0, vl);
        C7 = __riscv_vfmacc_vf_f16m2(C7, Br[bB + 7], A0, vl);

        bA += 32;
        bB += 8;
    }

    if (beta == 0.0f) {
        __riscv_vse32_v_f32m4(&Crref(0, 0), __riscv_vfwcvt_f_f_v_f32m4(C0, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 1), __riscv_vfwcvt_f_f_v_f32m4(C1, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 2), __riscv_vfwcvt_f_f_v_f32m4(C2, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 3), __riscv_vfwcvt_f_f_v_f32m4(C3, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 4), __riscv_vfwcvt_f_f_v_f32m4(C4, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 5), __riscv_vfwcvt_f_f_v_f32m4(C5, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 6), __riscv_vfwcvt_f_f_v_f32m4(C6, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 7), __riscv_vfwcvt_f_f_v_f32m4(C7, vl), vl);
    } else {
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 0), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C0, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 0), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 1), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C1, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 1), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 2), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C2, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 2), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 3), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C3, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 3), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 4), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C4, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 4), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 5), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C5, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 5), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 6), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C6, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 6), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 7), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C7, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 7), c, vl); }
    }
}

void ukernel_32x4_fp16(
    int mr, int nr, int kc,
    const fp16_t *Ar, const fp16_t *Br, float *Cr,
    float beta, int ldA, int ldB, int Clda)
{
    int pr, bA = 0, bB = 0;
    size_t vl = __riscv_vsetvl_e16m2(32);  
    const fp16_t zero = (fp16_t)0;

    vfloat16m2_t A0;
    vfloat16m2_t C0, C1, C2, C3;

    C0 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C1 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C2 = __riscv_vfmv_v_f_f16m2(zero, vl);
    C3 = __riscv_vfmv_v_f_f16m2(zero, vl);

    for (pr = 0; pr < kc; pr++) {
        A0 = __riscv_vle16_v_f16m2((const _Float16 *)&Ar[bA], vl);

        C0 = __riscv_vfmacc_vf_f16m2(C0, Br[bB + 0], A0, vl);
        C1 = __riscv_vfmacc_vf_f16m2(C1, Br[bB + 1], A0, vl);
        C2 = __riscv_vfmacc_vf_f16m2(C2, Br[bB + 2], A0, vl);
        C3 = __riscv_vfmacc_vf_f16m2(C3, Br[bB + 3], A0, vl);

        bA += 32;
        bB += 4;
    }

    if (beta == 0.0f) {
        __riscv_vse32_v_f32m4(&Crref(0, 0), __riscv_vfwcvt_f_f_v_f32m4(C0, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 1), __riscv_vfwcvt_f_f_v_f32m4(C1, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 2), __riscv_vfwcvt_f_f_v_f32m4(C2, vl), vl);
        __riscv_vse32_v_f32m4(&Crref(0, 3), __riscv_vfwcvt_f_f_v_f32m4(C3, vl), vl);
    } else {
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 0), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C0, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 0), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 1), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C1, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 1), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 2), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C2, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 2), c, vl); }
        { vfloat32m4_t c = __riscv_vle32_v_f32m4(&Crref(0, 3), vl);
          c = __riscv_vfadd_vv_f32m4(c, __riscv_vfwcvt_f_f_v_f32m4(C3, vl), vl);
          __riscv_vse32_v_f32m4(&Crref(0, 3), c, vl); }
    }
}

