#ifndef MODEL_LEVEL_H
#define MODEL_LEVEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef A57
  #include "archs/ARM_A57.h"
#elif A78AE
  #include "archs/ARM_A78AE.h"
#elif A72
  #include "archs/ARM_A72.h"
#elif K1
  #include "archs/K1.h"
#elif CARMEL
  #include "archs/NVIDIA_CARMEL.h"
#elif C906
  #include "archs/XUANTIE_C906.h"
#elif C910
  #include "archs/XUANTIE_C910.h"
#endif

#ifndef min
  #define min(a,b)     ( (a) > (b) ? (b) : (a) )
#endif

#ifndef max
  #define max(a,b)     ( (a) < (b) ? (b) : (a) )
#endif

void get_optim_mc_nc_kc(int th, int dataSize, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc);

#endif
