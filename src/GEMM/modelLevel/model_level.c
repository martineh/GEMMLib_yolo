//  Purpose
//    Estimate the dimension of the panels that will fit into a given level of the cache hierarchy following the
//    principles in the paper "Analytical modeling is enough for enough for high performance BLIS" by 
//    T. M. Low et al, 2016
//
//  Inputs:
//     NL:    Number of sets
//     CL:    Bytes per line
//     WL:    Associativity degree
//     (m,n): Dimensions of block in higher level of cache
//     Sdata: Bytes per element (e.g., 8 for FP64)
//
//  Output
//     k:     Determines that a block of size k x n stays in this level of the cache
//
//  Rule of thumb: subtract 1 line from WL (associativity), which is dedicated to the
//  operand that is "invariable" in this level, and distribute the rest between the two
//  other operands proportionaly to the ratio m/n
//
//  For example, with the conventional algorithm B3A2B1C0 and the L1 cache,
//  1 line is dedicated to Cr (invariable in loop L6) while the remaining lines are distributed
//  between Ar (mr x kc) and Br (kc x nr) proportionally to the ratio mr/nr.
//  Note that nc, in the ratio denominator, is associated with Br, the operand that resides in the L1 cache.
//  This is then leverated to calculate kc
//
//  Similarly, for the L2 cache, 1 line is dedicated to Br (invariable in loop L5)
//  while the remaining lines are distributed between C' (mc x nr) and Ac (mc x kc) proportionally to the ratio nr/kc.
//  Note that kc, in the ratio denominator, is associated with Ac, the operand that resides in the L2 cache.
//  This is then leveraged to calculate mc
//
//  Finally, for the L3 cache, 1 line is dedicated to Ac (invariable in loop L4)
//  while the remaining lines are distributed between Cc (mc x nc) and Bc (kc x nc) proportionally to the ratio mc/kc.
//  Note that kc, in the ratio denominator, is associated with Bc, the operand that resides in the L3 cache.
//  This is then leveraged to calculate nc
//

#include "model_level.h"

int model_level(int NL, int CL, int WL, int dataSize, int m, int n) { 
  int k, CAr, CBr;

  if (WL==2)
    k = NL * CL / (2.0 * m * dataSize);
  else {
     CAr = floor( ( (float)WL - 1.0 ) / (1.0 + (float)n / (float)m) ); //Lines of each set for Ar 
     if (CAr==0) { // Special case
       CAr = 1.0;
       CBr = WL - 2;
       //k = CBr * NL * CL / (n * dataSize);
     } else {
       CBr = ceil( ( (float)n / (float)m ) * (float)CAr ); //Lines of each set for Br
       //CBr = max(WL - 1 - CAr, CBr);
       //k = CAr * NL * CL / (m * dataSize);
     }
     k = CBr * NL * CL / (n * dataSize);
  }

   return k;
}



void get_optim_mc_nc_kc(int th, int dataSize, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc) {
  *kc = model_level(NL1, CL1, WL1, dataSize, mr, nr);
  //*kc = *kc / 16 * 16; //KR vlen for sdot

  #if defined(CARMEL) 
  if (th > 1) *kc = 512;
  #elif defined(A57) || defined(A72)
  if (th > 1) *kc = 640;
  #elif defined(K1)
  if (th > 1) *kc = 512;
  #elif defined(C910)
  if (th > 1) *kc = 512;
  #endif
  *kc = min(k, *kc);
  
  *mc = model_level(NL2, CL2, WL2, dataSize, nr, *kc);
  *mc = *mc / mr * mr;
  #if defined(A78AE) || defined(CARMEL)
  if (th > 1) *mc = 256;
  #elif defined(A57) || defined(A72)
  if (th > 1) *mc = 120;
  #elif defined(K1)
  if (th > 1) *mc = 512;
  #elif defined(C910)
  if (th > 1) *mc = 256;
  #endif
  *mc = min(m, *mc);

  *nc = model_level(NL3, CL3, WL3, dataSize, *mc, *kc);
  *nc = *nc / nr * nr;
  #if defined(A78AE) || defined(CARMEL)
  if (th > 1) *nc = 1024;
  #elif defined(A57) || defined(A72)
  if (th > 1) *nc = 3072;
  #elif defined(K1)
  if (th > 1) *nc = 1024;
  #elif defined(C910)
  if (th > 1) *nc = 1024;
  #endif
  *nc = min(n, *nc);


}

