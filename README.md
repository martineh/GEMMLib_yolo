

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

Once the library has been successfully compiled, it can be installed locally within the current directory by executing the following command:


```bash
make install
```

## 1. Run Tests

## 2. Running GEMM Experiments

To run the experiments, a configuration file is provided containing the matrix dimensions corresponding to the convolution operations performed by YOLO after applying the **im2col transformation**.

However, to evaluate other matrix multiplication scenarios, a new configuration file can be created by specifying different matrix dimensions following the provided file format.

The executable generates a **CSV file** containing the experimental results. Therefore, the desired output filename must be specified as an argument when running the executable.

**Note:** Make sure that the output directory already exists before running the experiments, as it will **not be created automatically**. If the directory does not exist, the execution will fail.

To run the experiments using the provided YOLO matrix dimensions, execute:


```bash
./gemm.bin gemm_study/yolo.dat output/yolo.csv
```

**Note:** The number of threads used during execution can be configured directly in the `gemm.bin` file by modifying the value of the `TH` variable.


