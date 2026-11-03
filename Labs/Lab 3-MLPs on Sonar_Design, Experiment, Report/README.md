# Lab 3: MLPs on Sonar: Design, Experiment, Report

**COMP 395 – Deep Learning**

In Lab 2 you built one sigmoid neuron from scratch and trained it on breast cancer data. In this lab you go past a single neuron. You build several **multilayer perceptrons (MLPs)** with `nn.Sequential`, choose your own activation functions, loss functions, and optimizers from the PyTorch documentation, and run each one as a tracked experiment in **Weights & Biases (W&B)**. The data comes from sonar: telling underwater **mines** (metal cylinders) from **rocks** using the energy of reflected sonar signals.

## Learning Objectives

By completing this lab, you will:

1. Learn a new PyTorch tool (`nn.Sequential`) from its documentation
2. Choose and construct activation functions, losses, and optimizers yourself
3. Explain why a nonlinearity between layers is required, and test for it
4. Run a controlled set of experiments, changing one thing at a time
5. Track, compare, and share experiments with Weights & Biases
6. Compare deep learning against a strong classical baseline (Random Forest)
7. Write a short, evidence-based experiment report built on W&B plots

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. **Create a free W&B account.** Go to <https://wandb.ai/site/research> and sign up with your **@oxy.edu** email. W&B is free for students and academic research; the free tier covers everything in this lab.
4. **Log in once per machine:**
   ```
   wandb login
   ```
   Paste the API key from <https://wandb.ai/authorize> when it asks. (In Colab, run `import wandb; wandb.login()` in a cell instead.)

   **Never commit your API key** to this repo or paste it into a notebook you push.

5. **In the class W&B organization?** Read `read-if-m2l2.pdf`. Your runs should go to your personal account.

## Step 1: Run the Linear Baseline (not graded)

```
python linear_baseline_wandb.py
```

This is your Lab 2 model (one sigmoid neuron, MSE loss, SGD, weights start at zero), now built with `nn.Sequential` and trained on Sonar. **It is complete; you do not edit it.** It follows the same structure as `binary_classification_wandb.py`, the W&B version of Lab 2 you have already seen (Parts 1–5, `parse_args()`, `train(..., run=None)`, `try`/`finally: run.finish()`), so read the two side by side. Two parts matter for the rest of the lab:

- **`linear_baseline(n_features)`** is an *experiment function*: it builds and returns the model, the loss function, and the optimizer. Every experiment you write has this shape.
- **The W&B lifecycle:**

| Code | What it does |
|------|--------------|
| `run = wandb.init(project=..., name=..., config={...})` | Starts a run; `config` records your choices |
| `run.log({"train/loss": ..., "val/acc": ...}, step=epoch)` | Sends metrics each epoch; each key becomes a chart |
| `run.log({"training_results": wandb.Image(fig)})` | Uploads a matplotlib figure |
| `run.summary["val_acc"] = ...` | Records final numbers (columns in the runs table) |
| `wandb.Artifact(...)` + `run.log_artifact(...)` | Saves the trained model, versioned |
| `try: ... finally: run.finish()` | Always closes the run, even if training crashes |

The script prints a link to the run. Open it and find the `train/loss`, `val/acc`, `training_results`, and `learned_weights` panels. The baseline gets about **81% validation accuracy**. Every MLP you build is measured against that number.

## Step 2: Read the Documentation

You will build your networks with tools nobody has shown you yet. Read these pages and, **with your partner**, answer the questions below. (Discuss them in class; they are not graded, but you will need the answers to write your experiments.)

- [`torch.nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
- [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [`torch.optim`](https://pytorch.org/docs/stable/optim.html)

**Documentation questions:**
1. What does `nn.Sequential` accept as arguments, and in what order does it call the modules you give it?
2. Given a model built with `nn.Sequential`, how do you get its first layer? (The baseline uses this.)
3. Name two activation functions other than ReLU. What range of values does each output?
4. For one optimizer you use: what arguments does it take, and what are their default values?

## Step 3: Design and Run Your Experiments

Open `sonar_mlp.py`. **You write Part 1 only.** Each experiment is a function:

```python
def experiment1(n_features):
    """TODO: one sentence: what this experiment is and what question it tests."""
    model = TODO
    loss_fn = TODO
    optimizer = TODO
    return model, loss_fn, optimizer
```

| You build | Requirement |
|-----------|-------------|
| `model` | An `nn.Sequential` that maps a `(B, 60)` batch to `B` probabilities. At least one hidden layer, a nonlinear activation between layers, and a sigmoid at the end |
| `loss_fn` | A loss function from `torch.nn` that works on probabilities |
| `optimizer` | An optimizer from `torch.optim`, built on **this model's** parameters, with the settings you choose |

Write at least **five** experiments. Across them, use **at least two different hidden-layer activations** and **at least two different optimizers**. Rename each function descriptively (e.g. `two_layer_relu_adam`); the name becomes the run name in W&B. Replace each `TODO` docstring with what the experiment tests. Then list your functions in `EXPERIMENTS`.

Parts 2–5 are provided. They train each experiment, record everything you chose (layer count, activations, loss, optimizer and its settings, parameter count) in the run's config, and log one W&B run per experiment.

```
python sonar_mlp.py                                  # run every experiment
python sonar_mlp.py --experiment two_layer_relu_adam # run just one
python sonar_mlp.py --help                           # all options
```

Plan your experiments so each one changes **one thing** from another: depth, width, activation, loss, optimizer, learning rate, dropout, and so on.

## Step 4: The Reality Check (`comparison.py`)

Train a **Random Forest** (`sklearn.ensemble.RandomForestClassifier`) on the same split (`load_data()` from `sonar_data.py`) and log it to the **same W&B project** as a run named `random_forest` in group `baseline`. This time you write the W&B code yourself, following `linear_baseline_wandb.py`: `run = wandb.init(...)`, `run.summary["val_acc"]`, and `try`/`finally: run.finish()`. The top of `comparison.py` lists what the file needs.

> **Using AI for this part:** It is OK to use an LLM to write `comparison.py`, but give it context first: paste `test_comparison.py`, `sonar_data.py`, and `linear_baseline_wandb.py` into the chat, or open this folder in Codex / Claude Code. Include your transcript and be ready to explain the code verbally.

## Testing Your Code

The tests **do not need a W&B login**. They set `WANDB_MODE=disabled` and call `train()` without a run. They train each experiment with the default settings (100 epochs, batch size 1, seed 0).

```
python -m pytest test_sonar_mlp.py -v
python -m pytest test_comparison.py -v
```

Or run one group at a time:

```
python -m pytest test_sonar_mlp.py::TestExperiments -v
```

| Test group | What it checks |
|------------|----------------|
| `TestExperiments` | 5+ experiments, each returning `(model, loss_fn, optimizer)`; model built with `nn.Sequential`, outputs probabilities of shape `(B,)` or `(B, 1)`, has a hidden layer; a `torch.nn` loss that works on probabilities; a `torch.optim` optimizer on this model's parameters; real docstrings; no two experiments identical |
| `TestChoices` | At least 2 different hidden-layer activations and 2 different optimizers across your experiments |
| `TestNonlinearity` | No model is secretly linear (stacked `nn.Linear` layers with no activation) |
| `TestTraining` | Every experiment trains without NaNs |
| `TestPerformance` | Your best experiment reaches **≥ 85%** final validation accuracy (linear baseline: ~81%) |
| `test_comparison.py` | Random Forest trained on `sonar_data`, logged to W&B, described in comments, runs without errors |

## Step 5: Experiment Report (40 points)

Keep it short: **no more than one page of text**, plus **a few plots from W&B**. Commit it in a folder called `report/` containing `report.tex`, `report.pdf`, and your images.

**Plots.** Export them from W&B (each panel's menu on wandb.ai, or a screenshot); no standalone matplotlib plots. Each plot should show **several of your models at once**, overlaid in one panel. At minimum:

1. **`train/loss`** for several of your experiments in one panel
2. **`val/loss`** (or `val/acc`) for the same experiments in one panel

**Text (one page at most).** Cover:

- What you tried, and what the loss curves show: which models learned fastest, which overfit (training loss keeps falling while validation loss rises), and which got stuck. Cite numbers from your W&B runs.
- How your best MLP compared with the linear baseline (~81%) and your Random Forest, and why you think that is.

Include the **link to your W&B project** and set the project's visibility to **Public** in its settings so the instructor can open it.

New to LaTeX? See `overleaf-tutorial.pdf`.

## Files

| File | Description |
|------|-------------|
| `linear_baseline_wandb.py` | **Complete example**: one experiment function + full W&B logging. Run it, read it, do not edit it |
| `sonar_mlp.py` | **Your code goes here**: your experiment functions (Part 1) |
| `comparison.py` | **Your code goes here**: Random Forest + W&B logging (Step 4) |
| `sonar_data.py` | Data loading shared by every script (do not modify) |
| `sonar.csv` | UCI Sonar dataset: 208 samples, 60 features, last column `M` (mine) or `R` (rock) |
| `test_sonar_mlp.py` | Automated tests for `sonar_mlp.py` (do not modify) |
| `test_comparison.py` | Automated tests for `comparison.py` (do not modify) |
| `requirements.txt` | Python dependencies |
| `read-if-m2l2.pdf` | For students in the class W&B organization: keeping runs in your own account |
| `overleaf-tutorial.pdf` | How to write your report in Overleaf |
| `report/` | **Your report goes here** (Step 5) |

## Troubleshooting W&B

| Problem | Fix |
|---------|-----|
| `wandb: ERROR api_key not configured` | Run `wandb login` again |
| No internet / W&B is down | `WANDB_MODE=offline python sonar_mlp.py`, then `wandb sync wandb/offline-run-*` once you are back online |
| Runs show up in a class/team workspace instead of your own | See `read-if-m2l2.pdf`. The scripts log to your personal username automatically; if they still don't, run `export WANDB_ENTITY=<your-username>` |
| Want to train without W&B | `python sonar_mlp.py --no_wandb` |
| Two runs merged into one | A run was not closed. Use `try`/`finally: run.finish()` in your own code in `comparison.py` |
| Too many junk runs in the project | Delete them from the runs table on wandb.ai, or filter before exporting charts |

## Grading

| Component | Points |
|-----------|--------|
| Experiment definitions (`TestExperiments`) | 15 |
| Your choices: 2+ activations, 2+ optimizers (`TestChoices`) | 10 |
| Every model is truly nonlinear (`TestNonlinearity`) | 5 |
| Every experiment trains (`TestTraining`) | 5 |
| Best experiment beats the linear baseline, ≥ 85% (`TestPerformance`) | 10 |
| Random Forest comparison (`test_comparison.py`) | 15 |
| Experiment report (`report/`: one page of text + W&B plots overlaying the loss of several models + public W&B link), graded by hand | 40 |
| **Total** | **100** |

## Submission

Commit and push `sonar_mlp.py`, `comparison.py`, your `report/` folder (`.tex`, `.pdf`, W&B images), and any AI transcripts to the default branch of this repository. Classroom 50 runs the tests on every push and reports your score for the code; the report is graded by hand. You can push as many times as you like before the deadline.
