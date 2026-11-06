# COMP395 Workstation GPU Benchmark

## Overview

This benchmark evaluates the computational performance of
the COMP395 workstation equipped with two NVIDIA RTX 6000
Ada Generation GPUs.

**Benchmark Date:** 2026-11-05

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen Threadripper PRO 7965WX |
| CPU Cores / Threads | 24 / 48 |
| System Memory | 256 GB DDR5-4800 (documented configuration) |
| GPU Configuration | 2 × NVIDIA RTX 6000 Ada Generation |
| GPU Memory | 48 GB per GPU |
| Operating System | Windows 10 |

Windows previously reported 223.23 GiB of physical memory.
The difference from the documented 256 GB configuration
has not yet been independently verified.

## Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.14.9 |
| PyTorch | 2.15.0+cu132 |
| CUDA Runtime | 13.2 |
| NVIDIA Driver | 597.30 |
| Computation Precision | FP32 (highest) |

## Methodology

### Single-GPU Benchmark

Both GPUs independently executed FP32 matrix
multiplication workloads.

The following methods were used:

- CUDA Event timing
- GPU warm-up
- Explicit CUDA device and stream selection
- Seven measured iterations per matrix size
- Median execution time
- No matrix allocation inside the timed region

Matrix sizes:

- 1024 × 1024
- 2048 × 2048
- 4096 × 4096
- 8192 × 8192
- 16384 × 16384

### Dual-GPU Benchmark

Two independent workloads were executed simultaneously
using Python worker threads.

Each GPU operated on its own matrices.

Dual-GPU throughput was calculated using the combined
workload and wall-clock completion time.

This experiment evaluates independent task parallelism.

## Single-GPU Results

At matrix size 4096 × 4096:

| GPU | Effective Throughput |
|-----|----------------------|
| GPU 0 | 48.718 TFLOPS |
| GPU 1 | 51.682 TFLOPS |

Both GPUs completed the matrix multiplication
benchmarks successfully.

Detailed timing results are recorded in:

`single_gpu_results.csv`

## Dual-GPU Results

### Matrix Size: 8192 × 8192

| Run | Speedup |
|-----|---------|
| Run 1 | 2.040× |
| Run 2 | 2.034× |
| Average | 2.037× |

### Matrix Size: 16384 × 16384

| Run | Speedup |
|-----|---------|
| Run 1 | 1.990× |
| Run 2 | 2.034× |
| Average | 2.012× |

### Overall Stable Results

Average throughput speedup across the four
large-matrix measurements:

**2.024×**

The measured throughput scaling is approximately
2× for independent large-matrix workloads.

Values slightly above 2× are attributed to
measurement variability, not super-linear
hardware scaling.

## 4096 × 4096 Measurement Instability

The smaller dual-GPU benchmark produced unstable
single-GPU baselines.

Initial benchmark speedups:

- Run 1: 3.493×
- Run 2: 2.864×

Additional CUDA timing diagnostics produced:

- Diagnostic 1: 2.165×
- Diagnostic 2: 1.223×

The substantial variation indicates that these
measurements are unsuitable for drawing a reliable
dual-GPU scaling conclusion.

The underlying cause has not been established.

All measurements are preserved in:

- `dual_gpu_results.csv`
- `4096_timing_diagnostics.csv`

The 4096 results are excluded from the primary
scaling analysis and visualization.

## Limitations

This benchmark measures FP32 matrix multiplication
and independent GPU task throughput.

It does not establish:

- NVLink communication performance
- Combined GPU memory capacity
- Single-model multi-GPU acceleration
- Distributed neural network training performance
- End-to-end application performance

The results may also be affected by GPU clocks,
Windows WDDM scheduling, and system load.

Comparisons against the RTX 4060 Laptop GPU
should use matching workloads, precision settings,
and timing methodology.

## Conclusion

Both NVIDIA RTX 6000 Ada GPUs were successfully
detected and validated.

Single-GPU computation completed successfully
on both devices.

Dual-GPU execution demonstrated approximately
2× throughput scaling for independent large-matrix
workloads.

The workstation is ready for further COMP395
deep learning experiments.

## Benchmark Artifacts

| File | Description |
|------|-------------|
| `Workstation_GPU_Benchmark.ipynb` | Benchmark notebook |
| `single_gpu_results.csv` | Single-GPU measurements |
| `dual_gpu_results.csv` | Dual-GPU benchmark history |
| `4096_timing_diagnostics.csv` | Timing instability diagnostics |
| `gpu_benchmark_comparison.png` | Performance visualization |
| `Benchmark_Summary.md` | Experiment summary |

---

COMP395 — Special Topics in Computer Science: Deep Learning

GitHub Branch: `workstation`

Environment: `Workstation`
