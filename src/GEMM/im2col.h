void im2col(float *cols, int ld, float *in, 
		int batches, int channels, int height, int width,
                 int oheight, int owidth, int kheight, int kwidth, 
		 int vpadding, int hpadding, int vstride, int hstride,
                 int vdilation, int hdilation);
