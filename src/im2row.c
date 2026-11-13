#include "gemm_blis.h"

void im2row(IN_TYPE *rows, int ld, IN_TYPE *in,
	    int batch, int height, int width, int channel, int oheight, int owidth,
	    int kheight, int kwidth, int vpadding, int hpadding, int vstride, int
	    hstride, int vdilation, int hdilation)
{

  int b, x, y, row, kx, ix, ky, iy, c, col;
 
  for (b = 0; b < batch; b++)
    #pragma omp parallel for private (y, row, kx, ix, ky, iy, c, col)
    for (x = 0; x < oheight; x++)
      for (y = 0; y < owidth; y++) {
	row = b * oheight * owidth + x * owidth + y;
	for (kx = 0; kx < kheight; kx++) {
	  ix = vstride * x + vdilation * kx - vpadding;
	  if (0 <= ix && ix < height)
	    for (ky = 0; ky < kwidth; ky++) {
	      iy = hstride * y + hdilation * ky - hpadding;
	      if (0 <= iy && iy < width)
		for (c = 0; c < channel; c++) {
		  col = c * kheight * kwidth + kx * kwidth + ky;
		  rows[row * channel * kheight * kwidth + col] = in[((b * height + ix) * width + iy) * channel + c];
		}
	    }
	}
      }
}

