/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "inutils.h"
#include "colors.h"
#include "gemm_blis.h"
#include "cnn_layer.h"

#define dabs(a)      ( (a) > 0 ? (a) : -(a) )
#define min(a,b)     ( (a) > (b) ? (b) : (a) )
#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]

#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]

#define iCrow(a1,a2)  iC[ (a1)*(ldC)+(a2) ]
#define iCcol(a1,a2)  iC[ (a2)*(ldC)+(a1) ]

#define Cgrow(a1,a2) Cg[ (a1)*(ldC)+(a2) ]
#define Cgcol(a1,a2) Cg[ (a2)*(ldC)+(a1) ]

#define iCgrow(a1,a2) iCg[ (a1)*(ldC)+(a2) ]
#define iCgcol(a1,a2) iCg[ (a2)*(ldC)+(a1) ]

void im2row(IN_TYPE *rows, int ld, IN_TYPE *in,
            int batch, int height, int width, int channel, int oheight, int owidth,
            int kheight, int kwidth, int vpadding, int hpadding, int vstride, int
            hstride, int vdilation, int hdilation);

extern void   convert_row2col(OUT_TYPE *Ai, OUT_TYPE *Ao, int m, int n);
extern int    generate_matrix_fp32(int, size_t, size_t, float *, size_t);
extern int    generate_matrix_int8(int, size_t, size_t, int8_t  *, size_t);
extern int    generate_matrix_int16(int, size_t, size_t, int16_t *, size_t);
extern int    generate_matrix_int32(int, size_t, size_t, int32_t *, size_t);
extern void   get_optim_mc_nc_kc(int th, int dataSize, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc);
extern double dclock();


#ifdef defined(NQ_FP32) | defined(NQ_FP16)
  #define generate_matrix_W generate_matrix_fp32
#else
  #define generate_matrix_W generate_matrix_int8
#endif
    
#if defined(NQ_FP32) | defined(FQ_INT8_FP32) | defined(NQ_FP16)
  #define generate_matrix_O generate_matrix_fp32
#else
  #define generate_matrix_O generate_matrix_int32
#endif

void *my_aligned_alloc(size_t alignment, size_t size, int zero) {
    size_t request_size = size + alignment;
    char* buf = (char*)(zero ? calloc(1, request_size) : malloc(request_size));

    size_t remainder = ((size_t)buf) % alignment;
    size_t offset = alignment - remainder;
    char* ret = buf + (unsigned char)offset;

    // store how many extra bytes we allocated in the byte just before the
    // pointer we return
    *(unsigned char*)(ret - 1) = offset;

    return (void*)ret;
}

/* Free memory allocated with aligned_alloc */
void my_aligned_free(void* aligned_ptr) {
    int offset = *(((char*)aligned_ptr) - 1);
    free(((char*)aligned_ptr) - offset);
}


int main(int argc, char *argv[]) {
  
  char test;
  char variant[20];

  IN_TYPE  *A     = NULL; 
  IN_TYPE  *Aptr  = NULL;

  IN_TYPE  *B     = NULL; 
  IN_TYPE  *DEXT  = NULL; 
  OUT_TYPE *C     = NULL;
  OUT_TYPE *Cg    = NULL;

  PACK_TYPE *packA = NULL;

  OUT_TYPE alpha, beta;

  double t1, t2, time, btime, tmin, flops, GFLOPS, bGFLOPS;

  size_t i, j, nreps, visual, ldA, ldB, ldC;
  size_t m, n, k, mc, nc, kc, mmin, mmax, mstep, nmin, nmax, nstep, kmin, kmax, kstep; 
  int th;

  int Co, Wo, Ho, Wf, Hf, Ci, Hi, Wi;

  double errorthd = 1.0e-6;
  double error, nrm, tmp;
  
  FILE *fd_csv;
  unsigned int cnn_num = 1;
  unsigned int cnn_enable = 0;

  char orderA = 'C',
       orderB = 'C',
       orderC = 'C',
       transA = 'N',
       transB = 'N';

  int SALGORITHM, SLOOP;
  int MC, NC, KC, MR, NR, mr, nr, bMR, bNR;
  int mr_init, mr_limit, mr_inc;
  int nr_init, nr_limit, nr_inc;
  int bestOf = 0;
  int prepackAc = 1; //Prepacking weights
  int prepackBc = 0;
  int tformat;
  cnn_test_t *cnn_test;
  uk_intrinsic uk;

  alpha      = atof(argv[7]);
  beta       = atof(argv[8]);
  
  tmin       = atof(argv[9]);
  test       = argv[10][0];
  fd_csv     = fopen(argv[12], "w");
  th         = atoi(argv[13]);
  SALGORITHM = atoi(argv[14]);
  SLOOP      = atoi(argv[15]);
  MC         = atoi(argv[18]);
  NC         = atoi(argv[19]);
  KC         = atoi(argv[20]);

  if (strcmp(argv[11], "RESNET50") == 0) 
    cnn_test = new_cnn_test(RESNET50_V15);
  else if (strcmp(argv[11], "VGG16") == 0)
    cnn_test = new_cnn_test(VGG16);
  else {
    printf("ERROR: CNN Model not available.\n");
    exit(-1);
  }

  if (strcmp(argv[21], "NHWC") == 0)      tformat = NHWC;
  else if (strcmp(argv[21], "NCHW") == 0) tformat = NCHW;
  else { printf("ERROR: CNN Format unrecognized.\n"); exit(-1); }


  fprintf(fd_csv, "#l;MR;NR;m;n;k;mc;nc;kc;Gflops\n");
  
  printf(" ===========================================================================================================\n");
  printf(" |                                           %sRUN CONFIGURATION %s                                            |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" ===========================================================================================================\n");
  printf(" |  [*] Mode Selected: ");
  printf(" %sFamily%s                                                                             |\n", COLOR_BOLDWHITE, COLOR_RESET);
  //----------------------------------------------------------------------------------------------------------------------
  printf(" |  [*] SIMD Selected: ");
  printf(" %sARMv8%s                                                                              |\n", COLOR_BOLDWHITE, COLOR_RESET);
  //----------------------------------------------------------------------------------------------------------------------
  printf(" |  [*] Model        :  %s%-40s%s                                           |\n", COLOR_BOLDWHITE, argv[11], COLOR_RESET);
  printf(" |  [*] Format       :  %s%-40s%s                                           |\n", COLOR_BOLDWHITE, argv[21], COLOR_RESET);
  printf(" |  [*] Output       :  %s%-40s%s                                           |\n", COLOR_BOLDWHITE, argv[12], COLOR_RESET);
  //----------------------------------------------------------------------------------------------------------------------
  printf(" |  [*] Tmin         :  %s%-5.2f%s                                                                              |\n", COLOR_BOLDWHITE, tmin, COLOR_RESET);
  printf(" -----------------------------------------------------------------------------------------------------------\n");


  printf("\n"); 
  printf(" ===========================================================================================================\n");
  printf(" |                       %sDRIVER FOR THE EVALUATION OF LOWERING CONVOLUTION%s                                 |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" ===========================================================================================================\n");
  printf(" |   %sMR    NR       M       N      K     MC      NC     KC%s   |    %sTime      GMOPS       Error%s    |   %sPASS%s  |\n", COLOR_BOLDWHITE, COLOR_RESET, COLOR_BOLDWHITE, COLOR_RESET, COLOR_BOLDWHITE, COLOR_RESET);
  printf(" -----------------------------------------------------------------------------------------------------------\n");

   
  for (unsigned int cnn_i = 0; cnn_i < cnn_test->layers; cnn_i++) {
   
    Co = cnn_test->cnn_model[cnn_i].Co;
    Wo = cnn_test->cnn_model[cnn_i].Wo;
    Ho = cnn_test->cnn_model[cnn_i].Ho;
    Wf = cnn_test->cnn_model[cnn_i].Wf;
    Hf = cnn_test->cnn_model[cnn_i].Hf;
    Ci = cnn_test->cnn_model[cnn_i].Ci;
    Hi = cnn_test->cnn_model[cnn_i].Hi;
    Wi = cnn_test->cnn_model[cnn_i].Wi;

    if (tformat == NHWC) { 
      //Format NHWC
      m =           Co;
      n =  1 * Ho * Wo;
      k = Ci * Wf * Hf;
    } else {
      //Format NCHW
      m = 1 * Ho * Wo;
      n = Co; 
      k = Ci * Wf * Hf;
    }
    
    A     = (IN_TYPE  *)my_aligned_alloc(32, (size_t)m*(size_t)k*sizeof(IN_TYPE), 1);   
    B     = (IN_TYPE  *)my_aligned_alloc(32, (size_t)k*(size_t)n*sizeof(IN_TYPE), 1);   
    DEXT  = (IN_TYPE  *)my_aligned_alloc(32, (size_t)k*(size_t)n*sizeof(IN_TYPE), 1);   
    C     = (OUT_TYPE *)malloc((size_t)m*(size_t)n*sizeof(OUT_TYPE));

    if ( test=='T' ) Cg = (OUT_TYPE *)malloc((size_t)m*(size_t)n*sizeof(OUT_TYPE));

    ldA = m;
    generate_matrix_W( COLUMN_MAJOR, m, k, A, ldA );
    ldB = k;
    generate_matrix_W( COLUMN_MAJOR, k, n, B, ldB );
    ldC = m;
    generate_matrix_O( COLUMN_MAJOR, m, n, C, ldC );
	  
    if ( test=='T' )
        for ( j=0; j<n; j++ )
  	  for ( i=0; i<m; i++ ) 
	    Cgcol(i,j)=Ccol(i,j);

      MR=16; NR=8; //TODO: WARNING: FIX This values
      //best_uk_selector(&MR, &NR, &uk );
      // MODEL VALUES: MC, NC and KC
      int mc_tmp, nc_tmp, kc_tmp;
      if (SALGORITHM == B3A2C0)
        get_optim_mc_nc_kc(th, sizeof(IN_TYPE), m, n, k, MR, NR, &mc_tmp, &nc_tmp, &kc_tmp);
      else
        get_optim_mc_nc_kc(th, sizeof(IN_TYPE), m, n, k, NR, MR, &mc_tmp, &nc_tmp, &kc_tmp);

      if (MC != -1 && NC != -1 && KC != -1) {
        mc = (size_t)MC; 
        nc = (size_t)NC; 
        kc = (size_t)KC;
      } else {
        mc = (size_t)mc_tmp; 
        nc = (size_t)nc_tmp; 
        kc = (size_t)kc_tmp;
      }

      PACK_TYPE *Ac  = (PACK_TYPE *)malloc((mc + MR) * (kc + 16 ) * 2 * 16 * sizeof(PACK_TYPE));
      PACK_TYPE *Bc  = (PACK_TYPE *)malloc((nc + NR) * (kc + 16 ) * 2 * 16 * sizeof(PACK_TYPE));
      OUT_TYPE *Ctmp = (OUT_TYPE  *)malloc(MR * NR * 16 * sizeof(OUT_TYPE));
 
      if (prepackAc) {
        packA = (PACK_TYPE  *)my_aligned_alloc(32, (size_t)(m + MR)*(size_t)(k + 16)*2*sizeof(PACK_TYPE), 1);
	prepack_A(orderA, m, k, A, ldA, packA, mc, kc, MR); 
      }

      #ifndef RISCV 
        #if defined(Q_INT8_INT32) || defined(NQ_FP16)
        orderC = 'R';
        ldC = n;
        #endif
      #endif

      time  = 0.0; 
      t1    = dclock();
      nreps = 0;
      while ( time <= tmin ) {
        if (tformat == NHWC) {
	  //Format NHWC
	  im2row(DEXT, Ci * Hf * Wf, B, 1, Hi, Wi, Ci, Ho, Wo, Hf, Wf, 0, 0, 1, 1, 1, 1);
	  gemm_B3A2C0( orderA, orderB, orderC, transA, transB, m, n, k, alpha,
                       A, ldA, DEXT, ldB, beta, C, ldC,  Ac, Bc, mc, nc, kc,
                       MR, NR, SLOOP, Ctmp, uk, packA);
        } else {
          //Format NCHW
	  //TODO: This format, matrix A isn't the im2col matrix. So we can't apply prepacking. 
	  //im2col(DEXT, 1 * Ho * Wo, D, 1, Ci, Hi, Wi, Ho, Wo, Hf, Wf, 0, 0, 1, 1, 1, 1);
	  //gemm_B3A2C0( orderA, orderB, orderC, transA, transB, m, n, k, alpha,
                       //A, ldA, B, ldB, beta, C, ldC,  Ac, Bc, mc, nc, kc,
                       //MR, NR, SLOOP, Ctmp, uk, packA);
	  printf("NCHW No supported\n"); exit(-1);
	}	

        t2   = dclock();
        time = ( t2 > t1 ? t2 - t1 : 0.0 );
        nreps++;
      }
      time = time/nreps;
      flops   = 2.0 * m * n * k;
      GFLOPS  = flops / (1.0e+9 * time );

      #ifndef RISCV 
        #if defined(Q_INT8_INT32) || defined(NQ_FP16)
        OUT_TYPE *C_tmp  = (OUT_TYPE *)malloc((size_t)m*(size_t)n*sizeof(OUT_TYPE));
        convert_row2col(C, C_tmp, m, n);
        free(C);
        C = C_tmp;
        ldC = m;
        #endif
      #endif

      // Test result
      if ( test=='T' )
        error = gemm_validation(COLUMN_MAJOR, NO_TRANSPOSE, NO_TRANSPOSE, A, DEXT, m, n, k, alpha, beta, ldA, ldB, ldC, C, Cg);
      else 
        error = -1.0;
  
      printf(" |   %-2d   %2d    %6zu  %6zu %6zu %6zu %6zu %6zu   |  %8.2e %10.2e %10.2e   |", 
		   MR, NR, m, n, k, mc, nc, kc, time, GFLOPS, error );
	    
      if ( test=='T' )
        if ( error < errorthd )   
          printf("%s    OK   %s|", COLOR_BOLDGREEN, COLOR_RESET);   
        else
          printf("%s  ERROR  %s|", COLOR_BOLDRED, COLOR_RESET);   
        
      else
        printf("%s    -    %s|", COLOR_RED, COLOR_RESET);   
  
      printf("\n");
  
      if (prepackAc)
        my_aligned_free(packA);

      free(Ac);
      free(Bc);
      free(Ctmp);
    
    fprintf(fd_csv, "%d;%d;%d;%zu;%zu;%zu;%zu;%zu;%zu;%.2e\n", cnn_i, bMR, bNR, m, n, k, mc, nc, kc, GFLOPS);

    my_aligned_free(A);
    my_aligned_free(B);
    my_aligned_free(DEXT);
    free(C);

    if ( test=='T' ) free(Cg);
  

  }
  
  printf(" ===========================================================================================================\n");
  printf("\n");

  return 0;
}

