# COMP395 Workstation — Environment Snapshot

## Overview

This directory documents the hardware and software environment used for deep learning experiments on the COMP395 workstation.

The workstation environment is maintained separately from the original coursework environment.

- Repository: Stellaris-Gallery/Deep-Learning
- Branch: `workstation`
- Deployment Environment: `Workstation`
- Snapshot Date: November 5, 2026
- Status: Verified

## 1. Hardware Specifications

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen Threadripper PRO 7965WX |
| CPU Cores | 24 |
| CPU Threads | 48 |
| System Memory | 256 GB DDR5 |
| Memory Speed | 4800 MT/s |
| GPU Configuration | 2 × NVIDIA RTX 6000 Ada Generation |
| GPU Architecture | NVIDIA Ada Lovelace |
| GPU Memory | 48 GB GDDR6 ECC per GPU |
| Total GPU Memory | 96 GB (48 GB × 2) |
| Operating System | Windows 10 |

### Memory Information

The workstation is documented with a nominal memory configuration of 256 GB DDR5-4800.

Windows currently reports:

- OS-reported physical memory: 223.23 GiB
- Documented memory configuration: 256 GB
- Memory transfer rate: 4800 MT/s

The difference between the documented capacity and OS-reported memory has not yet been independently verified.

## 2. GPU Configuration

| GPU Index | GPU Model | Reported VRAM |
|-----------|-----------|---------------|
| GPU 0 | NVIDIA RTX 6000 Ada Generation | 47.99 GiB |
| GPU 1 | NVIDIA RTX 6000 Ada Generation | 47.99 GiB |

### NVIDIA Driver

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 597.30 |
| CUDA Driver Compatibility | 13.2 |
| CUDA Runtime (PyTorch) | 13.2 |

Both GPUs are recognized by NVIDIA-SMI and PyTorch.

## 3. Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.14.9 |
| PyTorch | 2.15.0+cu132 |
| NumPy | 2.5.4 |
| CUDA Runtime | 13.2 |
| uv | 0.13.7 |
| Package Management | uv |
| Development Environment | Visual Studio Code |
| Notebook Environment | Jupyter / IPyKernel |

Additional Python dependencies include:

- Matplotlib
- Pandas
- IPyKernel

Exact installed package versions are recorded in `requirements-workstation.txt`.

## 4. Environment Isolation

The COMP395 environment is installed on the workstation's D: drive.

Environment structure:

```text
D:\COMP395-Workstation\
├── tools\
│   ├── git\
│   └── uv\
├── python\
├── venv\
│   └── deeplearning\
├── cache\
│   └── uv\
└── repos\
    └── Deep-Learning\
```

This configuration keeps the COMP395 Python environment separate from the existing Unreal Engine development environment.

The Python environment and COMP395 dependencies can be managed independently without modifying the Unreal Engine project.

## 5. GPU Verification

### CUDA Detection

The following checks completed successfully:

| Test | Result |
|------|--------|
| PyTorch Import | PASS |
| CUDA Availability | PASS |
| GPU Count | 2 |
| GPU 0 Detection | PASS |
| GPU 1 Detection | PASS |
| GPU 0 CUDA Computation | PASS |
| GPU 1 CUDA Computation | PASS |
| Jupyter Kernel Execution | PASS |

### GPU Computation Test

A matrix multiplication test was executed independently on both GPUs.

Test configuration:

```python
import torch

for i in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{i}")

    a = torch.ones((512, 512), device=device)
    result = a @ a

    torch.cuda.synchronize(device)

    print(f"GPU {i}: {result[0, 0].item()}")
```

Expected result:

```text
GPU 0: 512.0
GPU 1: 512.0
```

Actual result:

```text
GPU 0: 512.0
GPU 1: 512.0
```

**Verification Status: PASS**

Both NVIDIA RTX 6000 Ada GPUs successfully executed CUDA workloads.

## 6. Reproducibility

The environment snapshot includes:

- `README.md` — Hardware and software documentation
- `requirements-workstation.txt` — Installed Python dependencies
- `gpu_inventory.csv` — NVIDIA GPU inventory

These files provide a reference for reproducing and comparing workstation experiments.

The dependency snapshot records installed packages but does not guarantee identical results across different operating systems or hardware configurations.

## 7. Benchmarking Plan

The workstation will be benchmarked against the original COMP395 laptop environment.

### Original Laptop

- CPU: Intel Core i9-13900H
- GPU: NVIDIA RTX 4060 Laptop GPU
- GPU Memory: 8 GB
- Python: 3.14.5
- PyTorch: 2.10.0+cu130

### Workstation

- CPU: AMD Ryzen Threadripper PRO 7965WX
- GPU: 2 × NVIDIA RTX 6000 Ada Generation
- GPU Memory: 48 GB per GPU
- System Memory: 256 GB DDR5-4800
- Python: 3.14.9
- PyTorch: 2.15.0+cu132

Benchmarking will evaluate GPU computation time and performance differences.

Results will be recorded separately from the original coursework.

## 8. Repository Organization

- `main`: Original COMP395 assignments, labs, and coursework.
- `workstation`: Workstation verification, benchmarks, and experiment results.

The original coursework files are preserved on the `main` branch.

Workstation-specific experiments and results are maintained on the `workstation` branch.

## 9. Security and Documentation

This repository is public.

Environment documentation contains only non-sensitive technical information.

The following information must not be committed:

- Authentication tokens or credentials
- Private keys
- Internal network addresses
- Confidential research data
- Unrelated workstation project files

---

**COMP395 — Special Topics in Computer Science: Deep Learning**

**Environment: Workstation**

**Initial GPU Verification: Successful**    