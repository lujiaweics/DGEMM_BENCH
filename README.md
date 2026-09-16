# DGEMM-Bench

A performance and correctness benchmark for Double Precision General Matrix Multiply (DGEMM) implementations.

## Overview

DGEMM-Bench provides a pluggable framework to test different DGEMM algorithms (e.g., naive triple-loop, loop reordering, blocking, vectorization, multi-threading) against a high-performance reference implementation (OpenBLAS). It automatically generates performance curves (GFLOPS) and verifies numerical consistency.

## Project Structure

- `include/`: Public interface headers.
- `src/`: Core implementation and driver logic.
- `scripts/`: Python plotting scripts.
- `bin/`: Compiled executables (generated).
- `data/`: CSV results and plots (generated).

## Prerequisites

- **C Compiler**: GCC (recommended) or Clang with C11 support.
- **BLAS Library**: [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) installed on your system.
- **Python 3**: For plotting performance curves.
- **Matplotlib**: Required for the plotting script.

## Installation & Usage

1. **Configure OpenBLAS path**:
   Edit the `Makefile` or pass variables directly:
   ```bash
   make OPENBLAS_ROOT=/path/to/openblas
   ```

2. **Run Benchmarks**:
   To run the naive `ijk` implementation:
   ```bash
   make ijk
   ```

3. **Plot Results**:
   Generate performance comparison plots:
   ```bash
   make plot
   ```

4. **Clean up**:
   ```bash
   make clean
   ```

## Adding Your Own Implementation

1. Create a new source file `src/my_dgemm_<name>.c`.
2. Implement `my_dgemm()` matching the signature in `include/dgemm_bench.h`.
3. Add your implementation name to the `IMPLS` list in the `Makefile`.
4. Run `make <name>` to test.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
