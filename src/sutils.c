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

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "gemm_blis.h"

void convert_row2col(OUT_TYPE *Ai, OUT_TYPE *Ao, int m, int n) {
  //Arow: Input  Matrix in row-major
  //Acol: Output Matrix in col-major

  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      Ao[j * m + i] = Ai[i * n + j];
}

void convert_col2row(OUT_TYPE *Ai, OUT_TYPE *Ao, int m, int n) {
  //Arow: Input  Matrix in col-major
  //Acol: Output Matrix in row-major

  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      Ao[i * n + j] = Ai[j * m + i];
}

void generate_matrix_fp32( int orderM, size_t m, size_t n, float *M, size_t ldM ) {
/*
 * Generate a matrix with random entries
 * m      : Row dimension
 * n      : Column dimension
 * M      : Matrix
 * ldM   : Leading dimension
 *
 */
  int i, j;
  
  if ( orderM==102 ) //OpenBLAS Notation 'C' == 102, colum-major
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mcol(i,j) = ((float) rand())/RAND_MAX + 1.0;
  else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mrow(i,j) = ((float) rand())/RAND_MAX + 1.0;
}


void generate_matrix_int8( int orderM, size_t m, size_t n, int8_t *M, size_t ldM ) {
  int i, j;
  
  if ( orderM==102 ) //OpenBLAS Notation 'C' == 102, colum-major
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mcol(i,j) = (j + i) % 32;
  else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mrow(i,j) = (j + i) % 32;
}

void generate_matrix_int16( int orderM, size_t m, size_t n, int16_t *M, size_t ldM ) {
  int i, j;
  
  if ( orderM==102 ) //OpenBLAS Notation 'C' == 102, colum-major
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mcol(i,j) = (j + i) % 32;
  else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mrow(i,j) = (j + i) % 32;
}

void generate_matrix_int32( int orderM, size_t m, size_t n, int32_t *M, size_t ldM ) {
  int i, j;
  
  if ( orderM==102 ) //OpenBLAS Notation 'C' == 102, colum-major
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mcol(i,j) = (j + i) % 32;
  else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        Mrow(i,j) = (j + i) % 32;
}

/*===========================================================================*/
void print_matrix( char *name, int orderM, size_t m, size_t n, float *M, size_t ldM )
{
/*
 * Print a matrix to standard output
 * name   : Label for matrix name
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 *
 */
  size_t i, j;

  if ( orderM==102 ) //OpenBLAS Notation 'C' == 102, column-major
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        printf( "%s[%zu,%zu] = %14.8e;\n", name, i, j, ((float) Mcol(i,j)) );
  else
    for ( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
        printf( "%s[%zu,%zu] = %14.8e;\n", name, i, j, ((float) Mrow(i,j)) );
}


/*===========================================================================*/
double dclock()
{
/* 
 * Timer
 *
 */
  struct timeval  tv;
  // struct timezone tz;

  gettimeofday( &tv, NULL );   

  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}
/*===========================================================================*/
