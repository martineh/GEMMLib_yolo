

# Software for optimization YOLOv26 on RISC-V Platforms.

This software is designed to optimize YOLO (You Only Look Once) models for RISC-V platforms, focusing on improving computational efficiency and inference performance. It is specifically tailored for RISC-V systems equipped with **256-bit vector processing units**, leveraging vector capabilities to accelerate YOLO operations and maximize hardware utilization.


## 1. GEMM and GEMV Library Compilation

The first step consists of compiling the library specifically designed for GEMM (General Matrix Multiplication) and GEMV (General Matrix-Vector Multiplication) operations.

Compilation options must be configured in the `Makefile.inc` file, including the corresponding paths to external libraries such as **OpenBLAS** or **BLIS**, if these are preferred over the custom implementation.

**Note:** FP16 support is currently available **only through our custom implementation**. External libraries are not supported for FP16 operations.

Once the configuration is complete, compile the library by running the following command from the corresponding directory:


```bash
make
```

