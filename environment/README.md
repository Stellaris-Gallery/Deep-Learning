# COMP395 — Reproducible Windows Environment

This deployment package rebuilds the course environment on a Windows x86-64 machine. It retains **Python 3.14** and **PyTorch 2.10.0 with CUDA 13.0**, matching the laptop's Python/PyTorch family. The reference Python patch version is **3.14.5**. It creates a fresh `.venv` without inheriting packages or paths from the laptop.

## Files

| File | Purpose |
|---|---|
| `requirements.in` | Direct dependencies identified from the course notebooks and scripts |
| `requirements.lock.txt` | Exact versions and download hashes for direct and transitive dependencies |
| `wheels/` | The existing Python 3.14 Windows Gensim build, its source, and provenance |
| `../Tools/setup_environment.ps1` | Creates `.venv`, installs the lockfile, verifies the environment, and registers its kernel |
| `../Tools/check_environment.py` | Checks imports, locked versions, library operations, and computation on each visible GPU |

The lockfile is specific to Windows x86-64 and standard CPython 3.14 (not a free-threaded build). It is not a portable Linux/macOS requirements file.

## Workstation prerequisites

- Install 64-bit **Python 3.14**, preferably 3.14.5 to match the reference interpreter. The `py -3.14` launcher must find it, or pass `-PythonExe` explicitly.
- Install an NVIDIA driver compatible with the CUDA 13.0 PyTorch wheel. Verify that `nvidia-smi` lists both RTX 6000 Ada cards. A separate CUDA Toolkit is not needed for these prebuilt wheels; building custom CUDA extensions would be a different task.
- Install Git and VS Code with the Python and Jupyter extensions.
- Allow several GB for the environment and installer temporary files, plus additional space for datasets. The first install downloads a large PyTorch wheel.

Sources: [Python virtual environments](https://docs.python.org/3/library/venv.html), [official PyTorch 2.10.0 installation commands](https://pytorch.org/get-started/previous-versions/#v2100).

## Deploy on the Sony workstation

Clone the repository if it is not already present:

```powershell
git clone https://github.com/CharlieWu0788/Deep-Learning.git C:\Dev\COMP395
Set-Location C:\Dev\COMP395
```

For an existing clone, first save any local work, then update `main`:

```powershell
git switch main
git pull --ff-only
```

Create the experiment branch once (use `git switch` without `-c` if it already exists):

```powershell
git switch -c codex/rtx6000ada-experiments
powershell -NoProfile -ExecutionPolicy Bypass -File .\Tools\setup_environment.ps1 -ExpectedGpuCount 2
```

`ExecutionPolicy Bypass` applies to this one PowerShell process. The script fails on missing dependencies, incompatible Python, or failed GPU checks; it does not silently fall back to CPU. It does not install a system driver, download course datasets, log in to W&B, or execute notebooks. Existing `.venv` directories must be isolated Python 3.14 environments; incompatible ones are rejected rather than deleted.

If Python is installed without the launcher:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Tools\setup_environment.ps1 -PythonExe C:\Python314\python.exe -ExpectedGpuCount 2
```

In VS Code, select **`.venv\Scripts\python.exe`** as the interpreter/kernel. The named kernel is **COMP395 (Python 3.14, CUDA 13.0)**. To start JupyterLab directly:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Run the setup again after updating the lockfile. The script installs pinned dependencies and verifies them, but it does not remove unrelated packages subsequently installed into `.venv`; rename the environment and rebuild for a fully clean installation.

## Validation

```powershell
.\.venv\Scripts\python.exe Tools\check_environment.py --expected-gpus 2
```

The check imports the coursework libraries, checks all locked versions and `pip check`, runs a small offline BERT forward/backward pass, and performs matrix multiplication, RNN/LSTM/GRU backward passes, CNN backward, and a torchvision CUDA operation **on every visible GPU**. It writes `.runtime/environment-report.json`, including Python, package versions, GPU names and memory, lockfile hash, and pass/fail results.

Two GPUs are checked independently. Their memory is not automatically combined; ordinary notebook code that selects `cuda` will use GPU 0. Use separate processes with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1` for independent experiments. Give each process distinct output/checkpoint and notebook paths so simultaneous runs cannot overwrite each other.

## Notebook and data coverage

The environment covers the Python coursework: Lecture 1–8, Assignments 1 and 3, Lab 1, Lab 2/3 scripts, seq2seq translation, and **`COMPAS Analysis - Student.ipynb`**. The two original reference notebooks, **`Compas Analysis.ipynb`** and **`Cox with interaction term and independent variables.ipynb`**, contain `%%R` cells and require a separate R/rpy2 setup. Their R packages include dplyr, ggplot2, gridExtra, survival, ggfortify, car, and pROC. That R setup is not installed or validated by this package.

- Run notebooks and training scripts with their own assignment directory as the working directory; many loaders use `./data` or local CSV/image paths.
- IMDB, GloVe, MNIST, and Hugging Face model downloads are not stored in Git. The corresponding notebooks download them on first use. Lecture 5's Gensim cache and the RNN assignment's GloVe cache are separate.
- Downloaded lecture `data/` directories, `.cache/`, `.runtime/`, `.venv/`, and model checkpoints are ignored by Git. COMPAS and Sonar CSV data and supplied images are included.
- The seq2seq `translate.py` script requires `translation_model.pt`: train first or transfer the checkpoint separately. Cloning the repository does not restore ignored model files.
- W&B authentication is per machine. Use `.venv\Scripts\wandb.exe login` when ready to track runs; do not commit API keys. For offline experiments use `$env:WANDB_MODE = 'offline'`; scripts that support `--no_wandb` can disable tracking.
- Saved notebook output is not live kernel state. Run prerequisite cells in order when starting a new kernel. Re-executing code does not automatically update manually written result tables; revise those after checking the new measurements.

Preserve the original settings for the first workstation run. Record the environment report, dataset split, seed, accuracy, and elapsed time before introducing optimization experiments. Use a fixed validation set to select settings; keep the test set out of that selection process.

## Updating the lockfile

Edit `requirements.in` deliberately. From the repository root, use a compatible uv installation to regenerate the Windows/Python 3.14 lock:

```powershell
uv pip compile environment\requirements.in --python .venv\Scripts\python.exe --find-links environment\wheels --no-emit-find-links --generate-hashes --output-file environment\requirements.lock.txt
```

Rebuild and rerun the checks before committing a new lock. Commit the environment files and scripts to `main`; keep workstation training outputs and optimization experiments on `codex/rtx6000ada-experiments`.
