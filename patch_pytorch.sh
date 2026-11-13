
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PyTorch RISC-V Patcher
#
# Replaces modified PyTorch source files and configures
# the libGEMM.so dependency in CMakeLists.txt.
#
# Expected structure:
#
# GEMMLib_yolo/
# ├── patch_pytorch.sh
# ├── aten/
# │   ├── CMakeLists.txt
# │   ├── CPUBlas.cpp
# │   ├── Activation.cpp
# │   ├── BlasKernel.cpp
# │   └── Unfold2d.cpp
# └── install/
#     └── lib/
#         └── libGEMM.so
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/aten"

echo "=========================================="
echo "         PyTorch RISC-V Patcher"
echo "=========================================="

# ------------------------------------------------------------
# 1. PyTorch source directory
# ------------------------------------------------------------

read -r -p "Enter PyTorch source directory: " PYTORCH_DIR

if [[ -z "$PYTORCH_DIR" || ! -d "$PYTORCH_DIR" ]]; then
    echo "ERROR: Invalid PyTorch source directory."
    exit 1
fi

PYTORCH_DIR="$(cd "$PYTORCH_DIR" && pwd -P)"

if [[ ! -f "$PYTORCH_DIR/aten/src/ATen/CMakeLists.txt" ]]; then
    echo "ERROR: PyTorch source tree not found."
    exit 1
fi

# ------------------------------------------------------------
# 2. GEMMLib installation directory
# ------------------------------------------------------------

read -r -p "Enter GEMMLib directory [$SCRIPT_DIR]: " GEMM_DIR

GEMM_DIR="${GEMM_DIR:-$SCRIPT_DIR}"

if [[ ! -d "$GEMM_DIR" ]]; then
    echo "ERROR: GEMMLib directory does not exist."
    exit 1
fi

GEMM_DIR="$(cd "$GEMM_DIR" && pwd -P)"
GEMM_LIB="$GEMM_DIR/install/lib/libGEMM.so"

if [[ ! -f "$GEMM_LIB" ]]; then
    echo "ERROR: libGEMM.so not found:"
    echo "       $GEMM_LIB"
    echo "Please compile and install GEMMLib first."
    exit 1
fi

echo ""
echo "PyTorch directory: $PYTORCH_DIR"
echo "GEMMLib library:   $GEMM_LIB"
echo "Patch directory:   $PATCH_DIR"
echo ""

# ------------------------------------------------------------
# 3. Files to patch
# ------------------------------------------------------------

FILES=(
    "CMakeLists.txt:aten/src/ATen/CMakeLists.txt"
    "CPUBlas.cpp:aten/src/ATen/native/CPUBlas.cpp"
    "Activation.cpp:aten/src/ATen/native/cpu/Activation.cpp"
    "BlasKernel.cpp:aten/src/ATen/native/cpu/BlasKernel.cpp"
    "Unfold2d.cpp:aten/src/ATen/native/cpu/Unfold2d.cpp"
)

# Validate all source and destination files
for entry in "${FILES[@]}"; do
    src="${entry%%:*}"
    dst="${entry#*:}"

    if [[ ! -f "$PATCH_DIR/$src" ]]; then
        echo "ERROR: Missing patch file: $PATCH_DIR/$src"
        exit 1
    fi

    if [[ ! -f "$PYTORCH_DIR/$dst" ]]; then
        echo "ERROR: Missing PyTorch file: $dst"
        exit 1
    fi
done

# ------------------------------------------------------------
# 4. Configure CMakeLists.txt
# ------------------------------------------------------------

echo "Configuring CMakeLists.txt..."

TEMP_CMAKE="$(mktemp)"
trap 'rm -f "$TEMP_CMAKE"' EXIT

python3 - "$PATCH_DIR/CMakeLists.txt" "$TEMP_CMAKE" "$GEMM_LIB" <<'PY'
import pathlib
import re
import sys

source, destination, gemm_lib = sys.argv[1:]

content = pathlib.Path(source).read_text()

# Find the existing libGEMM.so dependency
pattern = (
    r'list\s*\(\s*APPEND\s+'
    r'ATen_CPU_DEPENDENCY_LIBS\s+'
    r'[^\n)]*libGEMM\.so\s*\)'
)

replacement = (
    f'list(APPEND ATen_CPU_DEPENDENCY_LIBS "{gemm_lib}")'
)

content, count = re.subn(
    pattern,
    lambda match: replacement,
    content
)

if count != 1:
    sys.exit(
        f"ERROR: Expected exactly one libGEMM.so "
        f"dependency in CMakeLists.txt, found {count}."
    )

pathlib.Path(destination).write_text(content)

print(f"libGEMM.so path configured: {gemm_lib}")
PY

# ------------------------------------------------------------
# 5. Backup original PyTorch files
# ------------------------------------------------------------

BACKUP_DIR="$(mktemp -d "$PYTORCH_DIR/pytorch_patch_backup_XXXXXXXX")"

echo ""
echo "Creating backup: $BACKUP_DIR"

for entry in "${FILES[@]}"; do
    dst="${entry#*:}"

    mkdir -p "$BACKUP_DIR/$(dirname "$dst")"
    cp -p "$PYTORCH_DIR/$dst" "$BACKUP_DIR/$dst"
done

# ------------------------------------------------------------
# 6. Apply patch
# ------------------------------------------------------------

echo ""
echo "Applying PyTorch patch..."

for entry in "${FILES[@]}"; do
    src="${entry%%:*}"
    dst="${entry#*:}"

    if [[ "$src" == "CMakeLists.txt" ]]; then
        cp "$TEMP_CMAKE" "$PYTORCH_DIR/$dst"
    else
        cp "$PATCH_DIR/$src" "$PYTORCH_DIR/$dst"
    fi

    echo "  [OK] $dst"
done

# ------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------

echo ""
echo "=========================================="
echo " PyTorch patch applied successfully!"
echo "=========================================="
echo "PyTorch: $PYTORCH_DIR"
echo "libGEMM: $GEMM_LIB"
echo "Backup:  $BACKUP_DIR"
echo ""
echo "PyTorch is ready for compilation."
echo "=========================================="

