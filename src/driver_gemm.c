
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

extern void   convert_row2col(OUT_TYPE *Ai, OUT_TYPE *Ao, int m, int n);
extern int    generate_matrix_fp32(int, size_t, size_t, float *, size_t);
extern int    generate_matrix_int8(int, size_t, size_t, int8_t  *, size_t);
extern int    generate_matrix_int16(int, size_t, size_t, int16_t *, size_t);
extern int    generate_matrix_int32(int, size_t, size_t, int32_t *, size_t);
extern void   get_optim_mc_nc_kc(int th, int dataSize, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc);
extern double dclock();


#if defined(NQ_FP32) | defined(NQ_FP16)
  #define generate_matrix_W generate_matrix_fp32
#else
  #define generate_matrix_W generate_matrix_int8
#endif
    
#if defined(NQ_FP32) | defined(NQ_FP16)
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
  OUT_TYPE *C     = NULL;
  OUT_TYPE *Cg    = NULL;

  PACK_TYPE *packA = NULL;

  OUT_TYPE alpha, beta;

  double t1, t2, time, btime, tmin, flops, GFLOPS, bGFLOPS;

  size_t i, j, nreps, visual;
  size_t mc, nc, kc, mmin, mmax, mstep, nmin, nmax, nstep, kmin, kmax, kstep; 
  int th;

  double errorthd = 1.0e-6;
  double error, nrm, tmp;
  
  FILE *fd_csv;
  testConfig_t* testConf = NULL;
  unsigned int cnn_num = 1;
  unsigned int cnn_enable = 0;

  char orderA = 'C',
       orderB = 'C',
       orderC = 'C',
       transA = 'N',
       transB = 'N';

  int SALGORITHM, SLOOP;
  int MC, NC, KC, MR, NR, mr, nr, bMR, bNR;
  int m, n, k, ldA, ldB, ldC;
  int mr_init, mr_limit, mr_inc;
  int nr_init, nr_limit, nr_inc;
  
  int bestOf = 0;

  uk_intrinsic uk;

  alpha      = atof(argv[7]);
  beta       = atof(argv[8]);
  
  tmin       = atof(argv[9]);
  test       = argv[10][0];
  testConf   = new_CNN_Test_Config(argv[11]);
  cnn_num    = testConf->cnn_num;
  fd_csv     = fopen(argv[12], "w");
  th         = atoi(argv[13]);
  SALGORITHM = atoi(argv[14]);
  SLOOP      = atoi(argv[15]);

  MC         = atoi(argv[18]);
  NC         = atoi(argv[19]);
  KC         = atoi(argv[20]);

  
  fprintf(fd_csv, "#l;m;n;k;Gflops:time\n");
  
  printf(" =========================================================================\n");
  printf(" |                        %sRUN CONFIGURATION%s                              |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" =========================================================================\n");
  printf(" |  [*] Dataset      :  %s%-30s%s                   |\n", COLOR_BOLDWHITE, argv[11], COLOR_RESET);
  printf(" |  [*] Output       :  %s%-30s%s                   |\n", COLOR_BOLDWHITE, argv[12], COLOR_RESET);
  printf(" |  [*] Tmin         :  %s%-5.2f%s                                            |\n", COLOR_BOLDWHITE, tmin, COLOR_RESET);
  printf(" -------------------------------------------------------------------------\n");


  printf("\n"); 
  printf(" =========================================================================\n");
  printf(" |                 %sDRIVER FOR THE EVALUATION OF GEMM%s                     |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" =========================================================================\n");
  printf(" |       %sM       N      K%s  |    %sTime      GFLOPS       Error%s    |   %sPASS%s  |\n", COLOR_BOLDWHITE, COLOR_RESET, COLOR_BOLDWHITE, COLOR_RESET, COLOR_BOLDWHITE, COLOR_RESET);
  printf(" -------------------------------------------------------------------------\n");

  
  for (unsigned int cnn_i = 0; cnn_i < cnn_num; cnn_i++) {
    
    m = testConf->cnn[cnn_i].h;
    n = testConf->cnn[cnn_i].l;
    k = testConf->cnn[cnn_i].b;
    
    A     = (IN_TYPE  *)my_aligned_alloc(32, (size_t)m*(size_t)k*sizeof(IN_TYPE), 1);   
    B     = (IN_TYPE  *)my_aligned_alloc(32, (size_t)k*(size_t)n*sizeof(IN_TYPE), 1);   
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


      //packA = (PACK_TYPE  *)my_aligned_alloc(32, (size_t)(m + 16)*(size_t)(k + 16)*2*sizeof(PACK_TYPE), 1);
      //prepack_A(orderA, m, k, A, ldA, packA, YOLO_MC, YOLO_KC, YOLO_MR);

      time  = 0.0; 
      t1    = dclock();
      nreps = 0;
      while ( time <= tmin ) {
        nreps++;

	gemm_rvv(&transA, &transB, &m, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC);
	//gemm_rvv(&transA, &transB, &m, &n, &k, &alpha, packA, &ldA, B, &ldB, &beta, C, &ldC);

        t2   = dclock();
        time = ( t2 > t1 ? t2 - t1 : 0.0 );
      }
      time = time/nreps;
      flops   = 2.0 * m * n * k;
      GFLOPS  = flops / (1.0e+9 * time );

      // Test result
      if ( test=='T' )
        error = gemm_validation('C', transA, transB, A, B, m, n, k, alpha, beta, ldA, ldB, ldC, C, Cg);
      else 
        error = -1.0;
  
      printf(" |   %6d  %6d %6d |  %8.2e %10.2e %10.2e   |", 
		   m, n, k, time, GFLOPS, error );
	    
      if ( test=='T' )
        if ( error < errorthd )   
          printf("%s    OK   %s|", COLOR_BOLDGREEN, COLOR_RESET);   
        else
          printf("%s  ERROR  %s|", COLOR_BOLDRED, COLOR_RESET);   
        
      else
        printf("%s    -    %s|", COLOR_RED, COLOR_RESET);   
  
      printf("\n");
  
    
    fprintf(fd_csv, "%d;%d;%d;%d;%.2e;%.4e\n", cnn_i, m, n, k, GFLOPS, time);

    my_aligned_free(A);
    my_aligned_free(B);
    free(C);

    //my_aligned_free(packA);

    if ( test=='T' ) free(Cg);
  

  }
  
  printf(" =========================================================================\n");
  printf("\n");

  return 0;
}

