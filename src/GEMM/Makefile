include Makefile.inc

#------------------------------------------
#| COMPILERS                              |
#------------------------------------------


ifeq ($(TVM), TVM_ENABLE)
  CC       = g++
  CLINKER  = g++
else
  CC       = gcc
  CLINKER  = gcc
endif

OPTFLAGS= -O3 -DCHECK -D$(QTYPE) -D$(PROCESSOR) -D$(IME)

ifeq ($(PROCESSOR), A57)
  arch=arm
else ifeq ($(PROCESSOR), A72)
  arch=arm
else ifeq ($(PROCESSOR), A78AE)
  arch=arm
else ifeq ($(PROCESSOR), CARMEL)
  arch=arm
else ifeq ($(PROCESSOR), C906)
  arch=riscv
else ifeq ($(PROCESSOR), C910)
  arch=riscv
else ifeq ($(PROCESSOR), C908)
  arch=riscv
else ifeq ($(PROCESSOR), K1)
  arch=riscv
else
  $(error Processor unsuported. Please, select a correct option in Makefile.inc.)
endif



ifeq ($(arch), arm)
    ifeq ($(PROCESSOR), A78AE)
        OPTFLAGS += -march=armv8.2-a+fp16+dotprod -DARMV8
    else ifeq ($(PROCESSOR), CARMEL)
        OPTFLAGS += -march=armv8.2-a+fp16 -DARMV8
    else
        OPTFLAGS += -march=armv8-a -DARMV8
    endif
else
    ifeq ($(PROCESSOR), C910)
	OPTFLAGS += -O2 -march=rv64imafdcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c910 -static -DRISCV
	CC        = riscv64-unknown-linux-gnu-gcc #Cross compiling
        CLINKER   = riscv64-unknown-linux-gnu-gcc
    else ifeq ($(PROCESSOR), C906)
    	OPTFLAGS += -g3 -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -static -DRISCV
	CC        = riscv64-unknown-linux-gnu-gcc #Cross compiling
        CLINKER   = riscv64-unknown-linux-gnu-gcc
    else
	OPTFLAGS += -O2 -march=rv64gcv_zfh -static -DRISCV
    endif
endif

#------------------------------------------

OPTFLAGS += -fopenmp -fPIC 

OBJDIR      = build
STATIC_LIB  = libGEMM.a
DINAMIC_LIB = libGEMM.so

#------------------------------------------
LIBS = -lm

LIBS_LINKER = $(LIBS)
INCLUDE = 
#------------------------------------------
SRC_CONV_FILES = $(wildcard ./src/*.c)
OBJ_CONV_FILES = $(patsubst ./src/%.c, $(OBJDIR)/%.o, $(SRC_CONV_FILES))

#SRC_QUANT_FILES = $(wildcard ./src/uk_quantize/*.c)
#OBJ_QUANT_FILES = $(patsubst ./src/uk_quantize/%.c, $(OBJDIR)/%.o, $(SRC_QUANT_FILES))

SRC_GEMM_FILES = $(wildcard ./src/gemm/*.c)
OBJ_GEMM_FILES = $(patsubst ./src/gemm/%.c, $(OBJDIR)/%.o, $(SRC_GEMM_FILES))

OBJ_FILES  = $(OBJDIR)/model_level.o $(OBJ_FP16) $(OBJ_CONV_FILES) $(OBJ_QUANT_FILES) $(OBJ_ASM_FILES) $(OBJ_GEMM_FILES) $(OBJ_CONVGEMM_FILES) 

ifeq ($(arch), arm)
  OBJ_FILES += $(OBJDIR)/uKernels_fp32.o
  OBJ_FILES += $(OBJDIR)/uKernels_int8_int32.o
else ifeq ($(PROCESSOR), K1)
  OBJ_FILES += $(OBJDIR)/uKernels_fp32.o
  OBJ_FILES += $(OBJDIR)/uKernels_int8_int32.o
else ifeq ($(PROCESSOR), C908)
  OBJ_FILES += $(OBJDIR)/uKernels_fp32.o
  OBJ_FILES += $(OBJDIR)/uKernels_int8_int32.o
endif

ifeq ($(PROCESSOR), A78AE)
    OBJ_FILES += $(OBJDIR)/uKernels_fp16.o
else ifeq ($(PROCESSOR), CARMEL)
    OBJ_FILES += $(OBJDIR)/uKernels_fp16.o
endif

all: $(OBJDIR)/$(STATIC_LIB) $(OBJDIR)/$(DINAMIC_LIB)

test: $(OBJDIR)/driver_gemm

install: all
	@mkdir -p install
	@mkdir -p install/lib
	@mkdir -p install/include
	install -m 644 $(OBJDIR)/$(STATIC_LIB) install/lib/
	install -m 644 $(OBJDIR)/$(DINAMIC_LIB) install/lib/
	install -m 644 src/gemm/gemm_blis.h install/include/

$(OBJDIR)/$(STATIC_LIB): $(OBJ_FILES)
	ar rc $@ $^

$(OBJDIR)/driver_gemm: $(OBJDIR)/inutils.o $(OBJDIR)/sutils.o $(OBJDIR)/driver_gemm.o
	$(CC) $(CFAGS) $(OPTFLAGS) -o $@ $^ -I./install/include/ -lGEMM -L/install/lib


$(OBJDIR)/$(DINAMIC_LIB): $(OBJ_FILES)
	$(CC) -fopenmp -shared -o $@ $^

$(OBJDIR)/%.o: ./src/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/gemm/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) -I $(LIBS)

$(OBJDIR)/%.o: ./src/uk_quantize/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) -I $(LIBS)

$(OBJDIR)/%.o: ./src/intrinsic_generator/ukernels/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) -I $(LIBS)

$(OBJDIR)/model_level.o: ./src/modelLevel/model_level.c 
	@mkdir -p $(OBJDIR)
	python3 src/intrinsic_generator/instrinsics_generator.py --arch $(arch) --op_b BROADCAST
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/inutils.o, $(OBJDIR)/sutils.o, $(OBJDIR)/driver_gemm.o: ./test/inutils.c ./test/sutils.c ./test/driver_gemm.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)


clean:
	rm $(OBJDIR)/* 

