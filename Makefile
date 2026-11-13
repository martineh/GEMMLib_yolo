include Makefile.inc

# ============================================================
# Configuration
# ============================================================

OBJDIR      := build
INSTALLDIR  := install
LIBDIR      := $(INSTALLDIR)/lib
INCDIR      := $(INSTALLDIR)/include

STATIC_LIB  := $(OBJDIR)/libGEMM.a
DYNAMIC_LIB := $(OBJDIR)/libGEMM.so

CC          := gcc-14
AR          := ar
ARFLAGS     := rcs
CLINKER     := gcc-14

# Architecture / project flags
OPTFLAGS    += -O3
OPTFLAGS    += -fopenmp
OPTFLAGS    += -fPIC
OPTFLAGS    += -march=rv64gcv_zfh_zvfh
OPTFLAGS    += -DRISCV

# Project-specific defines can come from Makefile.inc
OPTFLAGS    += -DCHECK
OPTFLAGS    += -D$(QTYPE)
OPTFLAGS    += -D$(PROCESSOR)
OPTFLAGS    += -D$(IME)
OPTFLAGS    += -D$(LINEAL_SELECT)

CPPFLAGS    := \
	-I./src \
	-I./src/GEMM \
	-I./src/GEMM/modelLevel \
	-I./src/GEMM/intrinsic_generator/ukernels \
	-I./src/GEMM/asm_generator/ukernels

CFLAGS      += $(OPTFLAGS)
LDFLAGS     += -fopenmp
LIBS        := -lm -fopenmp

ifeq ($(LINEAL_SELECT),OPENBLAS_ON)
    CPPFLAGS += -I$(OPENBLAS_PATH)/include
    LIBS     += -L$(OPENBLAS_PATH)/lib -lopenblas

else ifeq ($(LINEAL_SELECT),BLIS_ON)
    CPPFLAGS += -I$(BLIS_PATH)/include
    LIBS     += -L$(BLIS_PATH)/lib -lblis

endif

# ============================================================
# GEMM library sources
# ============================================================

SRC_GEMM_C := \
	$(wildcard src/GEMM/*.c) \
	$(wildcard src/GEMM/modelLevel/*.c) \
	$(wildcard src/GEMM/intrinsic_generator/ukernels/*.c) \
	$(wildcard src/GEMM/asm_generator/ukernels/*.c)

SRC_GEMM_S := \
	$(wildcard src/GEMM/asm_generator/ukernels/*.S)

OBJ_GEMM_C := $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRC_GEMM_C))
OBJ_GEMM_S := $(patsubst src/%.S,$(OBJDIR)/%.o,$(SRC_GEMM_S))

OBJ_GEMM := $(OBJ_GEMM_C) $(OBJ_GEMM_S)


# ============================================================
# Executable sources
# ============================================================

SRC_COMMON := \
	src/inutils.c \
	src/sutils.c \
	src/im2row.c \
	src/cnn_layer.c

OBJ_COMMON := $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRC_COMMON))

OBJ_GEMM_DRIVER := $(OBJDIR)/driver_gemm.o
OBJ_CONV_DRIVER := $(OBJDIR)/driver_convolution.o


# ============================================================
# Targets
# ============================================================

.PHONY: all lib test install clean distclean

all: lib test


lib: $(STATIC_LIB) $(DYNAMIC_LIB)


test: $(OBJDIR)/gemm.x


# ============================================================
# Static library
# ============================================================

$(STATIC_LIB): $(OBJ_GEMM)
	@mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $^


# ============================================================
# Dynamic library
# ============================================================

$(DYNAMIC_LIB): $(OBJ_GEMM)
	@mkdir -p $(@D)
	$(CLINKER) -shared -o $@ $^ $(LIBS)


# ============================================================
# Executables
# ============================================================

$(OBJDIR)/gemm.x: $(OBJ_COMMON) $(OBJ_GEMM_DRIVER) $(STATIC_LIB)
	@mkdir -p $(@D)
	$(CLINKER) $(OPTFLAGS) -o $@ \
		$(OBJ_COMMON) \
		$(OBJ_GEMM_DRIVER) \
		-L$(OBJDIR) -lGEMM \
		$(LIBS)


$(OBJDIR)/convolution.x: $(OBJ_COMMON) $(OBJ_CONV_DRIVER) $(STATIC_LIB)
	@mkdir -p $(@D)
	$(CLINKER) $(OPTFLAGS) -o $@ \
		$(OBJ_COMMON) \
		$(OBJ_CONV_DRIVER) \
		-L$(OBJDIR) -lGEMM \
		$(LIBS)


# ============================================================
# Compilation rules - C
# ============================================================

$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@


# ============================================================
# Compilation rules - Assembly
# ============================================================

$(OBJDIR)/%.o: src/%.S
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@


# ============================================================
# Install
# ============================================================

install: lib
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCDIR)

	install -m 644 $(STATIC_LIB) $(LIBDIR)/
	install -m 755 $(DYNAMIC_LIB) $(LIBDIR)/

	install -m 644 src/GEMM/gemm_blis.h $(INCDIR)/

	@echo ""
	@echo "GEMMLib installed in $(INSTALLDIR)"
	@echo "Libraries:"
	@echo "  $(LIBDIR)/libGEMM.a"
	@echo "  $(LIBDIR)/libGEMM.so"
	@echo ""


# ============================================================
# Clean
# ============================================================

clean:
	rm -rf $(OBJDIR)/*.o
	rm -rf $(OBJDIR)/GEMM

distclean:
	rm -rf $(OBJDIR)
	rm -rf $(INSTALLDIR)
