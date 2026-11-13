

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

## 2. Running GEMM Experiments

To run the experiments, a configuration file is provided containing the matrix dimensions corresponding to the convolution operations performed by YOLO after applying the **im2col transformation**.

However, to evaluate other matrix multiplication scenarios, a new configuration file can be created by specifying different matrix dimensions following the provided file format.

The executable generates a **CSV file** containing the experimental results. Therefore, the desired output filename must be specified as an argument when running the executable.

**Note:** Make sure that the output directory already exists before running the experiments, as it will **not be created automatically**. If the directory does not exist, the execution will fail.

To run the experiments using the provided YOLO matrix dimensions, execute:


```bash
chmod +x gemm.bin
./gemm.bin gemm_study/yolo.dat output/yolo.csv
```

**Note:** The number of threads used during execution can be configured directly in the `gemm.bin` file by modifying the value of the `TH` variable.

## 3. PyTorch Patching

To integrate the optimized GEMM and GEMV implementations into PyTorch, an automated patching script is provided. The modified PyTorch source files are located in the `aten/` directory of the GEMMLib_yolo project.

### 3.1. Cloning the PyTorch Repository

Before applying the modifications, the PyTorch source code must be downloaded by cloning the official repository:

```bash
git clone --recursive https://github.com/pytorch/pytorch.git
```

**Note:** The `--recursive` option ensures that all required Git submodules are also cloned.

### 3.2. Applying the Patch

Once the PyTorch repository has been cloned, the provided patching script can be used to integrate the optimized GEMM and GEMV implementations.

The script automatically replaces the corresponding PyTorch source files and updates the `CMakeLists.txt` configuration to link against the custom `libGEMM.so` library.

Before applying the patch, ensure that **GEMMLib has been successfully compiled and installed**.

From the `GEMMLib_yolo` directory, execute:

```bash
chmod +x patch_pytorch.sh
./patch_pytorch.sh
```

During execution, the script will request:

1. **PyTorch source directory:** The path to the cloned PyTorch repository where the modifications will be applied.
2. **GEMMLib installation directory:** The path to the GEMMLib project. By default, the script uses its own directory and expects the compiled library at `install/lib/libGEMM.so`. Press `Enter` to accept the default path.

The script automatically performs the following operations:

- Verifies that all required source files and the compiled `libGEMM.so` library exist.
- Updates the library path in the PyTorch `CMakeLists.txt` file.
- Creates backup copies of the original PyTorch files.
- Copies the modified files from `GEMMLib_yolo/aten/` to their corresponding locations within the PyTorch source tree.

**Note:** The original PyTorch files are backed up before being overwritten. Once the patch has been successfully applied, **PyTorch must be compiled from source** for the modifications to take effect.

## 4. PyTorch and YOLO26 Compilation

The final step consists of compiling **PyTorch with the previously applied modifications** and setting up **YOLO26** to use the optimized GEMM and GEMV implementations.

### 4.1. PyTorch Compilation

Once the patch has been successfully applied, PyTorch must be compiled from source to incorporate the optimized GEMM and GEMV routines.

The compilation process must be performed on the target RISC-V platform or using an appropriate cross-compilation environment, ensuring that the required dependencies and compiler configurations are correctly set up.

**Note:** The patched PyTorch implementation must be used instead of a precompiled version, as standard PyTorch installations do not include the custom GEMM and GEMV optimizations.

### 4.2. YOLO26 Installation

After successfully compiling and installing the modified PyTorch version, the next step is to install YOLO26 through the Ultralytics framework.

It is essential to ensure that YOLO26 uses the previously compiled PyTorch installation so that inference operations can benefit from the optimized GEMM and GEMV kernels.

**Note:** Installing Ultralytics may introduce additional dependencies. Make sure that the installation process does not replace the custom-built PyTorch version with a precompiled package.

Once both components are correctly installed, the system will be ready to execute YOLO26 inference using the optimized RISC-V implementation with support for 256-bit vector processing units.
