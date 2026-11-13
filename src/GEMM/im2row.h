void im2row(float *rows, int ld, float *in, int batch, 
	    int height, int width, int channel, int oheight, 
	    int owidth, int kheight, int kwidth, 
	    int vpadding, int hpadding, int vstride, int hstride, 
	    int vdilation, int hdilation, int TH);
