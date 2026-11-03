# Environment validation

Validated on **2026-11-03** using a freshly created, isolated Windows environment.

| Check | Result |
|---|---|
| Python | 3.14.5, 64-bit |
| PyTorch / torchvision | 2.10.0+cu130 / 0.25.0+cu130 |
| GPU tested | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB |
| Locked dependencies | 160 installed versions match; pip check passes |
| Gensim | Bundled Python 3.14 wheel imports and performs vector operations |
| Offline BERT | Small randomly initialized model completes forward/backward |
| GPU operations | Matrix multiplication, RNN/LSTM/GRU backward, CNN backward, torchvision NMS pass |
| Lab 2 existing tests | 31 passed |
| Lab 3 existing tests | 21 passed |
| Jupyter kernel | Six Lecture 1 code cells executed with zero errors; output saved to a temporary copy |

The original coursework notebooks were not overwritten. Lab 2 emitted four existing tensor-copy warnings. The notebook kernel emitted Windows event-loop and local TCP transport warnings but completed successfully.

The Sony workstation and its two RTX 6000 Ada GPUs have **not** been accessed or tested. Run setup with `-ExpectedGpuCount 2` there; the checker tests every visible GPU independently. This validation establishes installation and representative coursework compatibility, not a complete rerun of every notebook, reproduction of historical accuracy, dataset download availability, or R/rpy2 support.

Lockfile SHA-256: `b9d54a72ea500502c6e88cd76830555c5c80099798b30e3c32b5203c525d1537`.

The detailed local JSON report and pytest XML files are under `.runtime/` and are intentionally excluded from Git. Regenerate the report on the workstation to record its actual hardware and installation.
