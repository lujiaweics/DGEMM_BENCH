# ============================================================================
#  DGEMM-Bench -- Double Precision General Matrix Multiply Performance Benchmark
#
#  Reference (golden) implementation uses OpenBLAS CBLAS interface;
#  Tested implementations are pluggable via a unified my_dgemm() interface.
#
#  ---------------------------------------------------------------------------
#  Prerequisites: Compile and install OpenBLAS, then specify its location:
#      make OPENBLAS_ROOT=$HOME/openblas
#      make BLAS_LIB=/opt/openblas/lib/libopenblas.so BLAS_INC=/opt/openblas/include
#      make BLAS_LIB=-lopenblas BLAS_INC=/usr/include       # Use system library
#
#  Common Targets:
#      make ijk      Compile and run naive ijk implementation, output to data/output_ijk.csv
#      make run      Run all registered implementations
#      make plot     Plot GFLOPS curves using Python3 (executes 'run' first)
#      make all      Equivalent to 'run'
#      make clean    Remove executables, data, and plots
#
#  Note: Build rules depend on POSIX shell (Linux / macOS / WSL / MSYS2).
# ============================================================================

# ------------------------------- Configuration -------------------------------
OPENBLAS_ROOT ?= /home/garvey/workspace/OpenBLAS-0.3.34/build
BLAS_LIB      ?= $(OPENBLAS_ROOT)/lib/libopenblas.a
BLAS_INC      ?= $(OPENBLAS_ROOT)/include

CC     := gcc
PYTHON := python3

# Note: -D_POSIX_C_SOURCE=... is not needed here as src/timer.c 
# defines it internally before including system headers to ensure 
# POSIX extensions are available without interfering with MinGW headers.
CFLAGS  := -O3 -std=c11 -march=native -Wall -Wextra -Iinclude -I$(BLAS_INC)
LDFLAGS := -lpthread -lm

# ------------------------------- Experiment Parameters -----------------------
# Four parameters passed to driver: repeats, first size, last size, increment
NREPEATS := 3
NFIRST   := 48
NLAST    := 1500
NINC     := 48

# -------------------------------- Directories --------------------------------
BIN_DIR  := bin
DATA_DIR := data
PLOT_PY  := scripts/plot_dgemm.py

# Common source files independent of specific implementations
COMMON_SRC := src/driver.c src/ref_blas.c src/timer.c \
              src/random_matrix.c src/max_abs_diff.c

# --------------------------- Implementation List (Pluggable) -----------------
# Steps to add a new implementation:
#   1) Create src/my_dgemm_<name>.c exporting my_dgemm() matching dgemm_bench.h;
#   2) Append <name> to IMPLS below;
#   3) Copy the "ijk" rules below and replace "ijk" with your <name>.
#
# Note: Explicit rules are used for each implementation because driver 
#       parameters may vary (e.g., some require n to be a multiple of MR/NR).
IMPLS := ijk ikj

.PHONY: all run plot clean distclean $(IMPLS)

all: run

# ============================================================================
#  ijk -- Naive Triple-Loop (Baseline)
# ============================================================================
ijk: $(BIN_DIR)/dgemm_ijk.x | $(DATA_DIR)
	$(BIN_DIR)/dgemm_ijk.x $(NREPEATS) $(NFIRST) $(NLAST) $(NINC) \
		> $(DATA_DIR)/output_ijk.csv
	@echo "==> $(DATA_DIR)/output_ijk.csv"

$(BIN_DIR)/dgemm_ijk.x: $(COMMON_SRC) src/my_dgemm_ijk.c Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) -DIMPL_NAME='"ijk"' $(filter %.c,$^) \
		$(BLAS_LIB) -o $@ $(LDFLAGS)

ikj: $(BIN_DIR)/dgemm_ikj.x | $(DATA_DIR)
	$(BIN_DIR)/dgemm_ikj.x $(NREPEATS) $(NFIRST) $(NLAST) $(NINC) \
		> $(DATA_DIR)/output_ikj.csv
	@echo "==> $(DATA_DIR)/output_ikj.csv"

$(BIN_DIR)/dgemm_ikj.x: $(COMMON_SRC) src/my_dgemm_ikj.c Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) -DIMPL_NAME='"ikj"' $(filter %.c,$^) \
		$(BLAS_LIB) -o $@ $(LDFLAGS)

# ============================================================================
#  Run and Plot
# ============================================================================
run: $(IMPLS)

# CSV list corresponding to IMPLS, no changes needed when adding implementations
CSV_ALL := $(addprefix $(DATA_DIR)/output_,$(addsuffix .csv,$(IMPLS)))

plot: $(IMPLS)
	$(PYTHON) $(PLOT_PY) $(CSV_ALL) -o $(DATA_DIR)/dgemm_gflops.png

# ============================================================================
#  Directories and Cleanup
# ============================================================================
$(BIN_DIR) $(DATA_DIR):
	@mkdir -p $@

clean:
	rm -f $(BIN_DIR)/*.x $(DATA_DIR)/*.csv $(DATA_DIR)/*.png

distclean: clean
	rm -rf $(BIN_DIR) $(DATA_DIR)
