"""
MLP Experiments on Sonar, with Weights & Biases Tracking
COMP 395 – Deep Learning

YOUR CODE GOES IN PART 1. Parts 2–5 are provided, and follow the same
structure as binary_classification_wandb.py from Lab 2.

Run `python -m pytest test_sonar_mlp.py -v` to test your experiments.
Run `python sonar_mlp.py --help` for training options.
"""

import argparse
import functools
import os

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # save figures without opening windows
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import TensorDataset, DataLoader

from sonar_data import load_data

# Placeholder so the file imports before you finish. Replace every TODO below.
TODO = None


# =============================================================================
# PART 1: Your Experiments
# =============================================================================
#
# Each experiment is a function that BUILDS AND RETURNS three things:
#
#   model      an nn.Sequential that maps a (B, 60) batch to B probabilities
#              in [0, 1]. It needs at least one hidden layer and a nonlinear
#              activation between layers, and must end in a sigmoid.
#   loss_fn    a loss function object from torch.nn
#   optimizer  an optimizer object from torch.optim, built on THIS model's
#              parameters, with the learning rate (and any other settings)
#              you choose
#
# You decide the architecture, the activations, the loss, and the optimizer.
# Nothing is chosen for you: read the documentation and pick.
#
#   nn.Sequential:          https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
#   nn.Linear:              https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
#   Activation functions:   https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
#   Loss functions:         https://pytorch.org/docs/stable/nn.html#loss-functions
#   Optimizers:             https://pytorch.org/docs/stable/optim.html
#
# >>> EXAMPLE: see the linear_baseline() function in linear_baseline_wandb.py.
# >>> It is a complete, working experiment function: it builds an
# >>> nn.Sequential model, a loss function, and an optimizer, and returns all
# >>> three. Every experiment you write below has exactly the same shape.
#
# Requirements (the tests check these):
#   * at least 5 experiments, and no two identical
#   * across your experiments, at least 2 different hidden-layer activation
#     functions and at least 2 different optimizers
#   * replace each TODO docstring with one sentence: what the experiment is
#     and what question it tests
#   * rename the functions descriptively (e.g. two_layer_relu_adam); the
#     function name becomes the run name in W&B
#
# Good experiments change ONE thing at a time. Need more than five? Write
# another function and add it to EXPERIMENTS.


# Not sure how to start? Open linear_baseline_wandb.py and read linear_baseline().
def relu32_adam(n_features):
    """Test whether a 32-unit ReLU hidden layer with Adam improves on the linear baseline."""
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer


def tanh32_adam(n_features):
    """Change only the hidden activation from relu32_adam to test how Tanh affects learning."""
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer


def relu64_adam(n_features):
    """Widen relu32_adam's hidden layer to 64 units to test the effect of greater width."""
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer


def relu32_16_adam(n_features):
    """Add a 16-unit ReLU hidden layer to relu32_adam to test the effect of greater depth."""
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer


def relu32_sgd(n_features):
    """Replace Adam in relu32_adam with SGD at the same learning rate to compare optimizers."""
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer


# All variants use BCE and lr=0.001, changing one factor from relu32_adam.
EXPERIMENTS = [relu32_adam, tanh32_adam, relu64_adam, relu32_16_adam, relu32_sgd]


# =============================================================================
# PART 2: Data Loading (provided, see sonar_data.py)
# =============================================================================
#
# load_data() returns X_train (166, 60), X_test (42, 60), y_train, y_test,
# normalized with training statistics. Every script uses the same split.


# =============================================================================
# PART 3: Training Loop (provided)
# =============================================================================

def train(model, loss_fn, optimizer, X_train, y_train, n_epochs=100,
          batch_size=1, verbose=True, run=None, X_val=None, y_val=None, seed=0):
    """
    Train any model with any loss and optimizer you built.

    Args:
        model: your nn.Sequential
        loss_fn: your loss function
        optimizer: your optimizer (built on model.parameters())
        X_train: (m, n) training features
        y_train: (m,) training labels
        n_epochs: number of passes through the data
        batch_size: samples per gradient step
        verbose: print progress
        run: optional wandb run; if given, per-epoch metrics are logged to it
        X_val, y_val: optional held-out data, evaluated each epoch when logging
        seed: seeds the shuffling order

    Returns:
        losses: list of average training loss per epoch
    """
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in loader:
            # Forward pass
            y_hat = model(x_batch).view(-1)
            loss = loss_fn(y_hat, y_batch)

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)

        avg_loss = epoch_loss / len(y_train)
        losses.append(avg_loss)

        # Log this epoch to wandb (skipped when no run is passed, e.g. in tests)
        if run is not None:
            metrics = {
                "train/loss": avg_loss,
                "train/acc": accuracy(y_train, predict(model, X_train)),
            }
            if X_val is not None and y_val is not None:
                metrics["val/loss"] = evaluate_loss(model, loss_fn, X_val, y_val)
                metrics["val/acc"] = accuracy(y_val, predict(model, X_val))
            run.log(metrics, step=epoch)

        if verbose and epoch % 10 == 0:
            print(f'Epoch {epoch:3d}, Loss: {avg_loss:.4f}')

    return losses


# =============================================================================
# PART 4: Evaluation and Bookkeeping (provided)
# =============================================================================

@torch.no_grad()
def predict(model, X):
    """Predict 0/1 labels for every row of X at once."""
    model.eval()
    return (model(X).view(-1) >= 0.5).float()


@torch.no_grad()
def evaluate_loss(model, loss_fn, X, y):
    """Loss on a whole dataset at once."""
    model.eval()
    return loss_fn(model(X).view(-1), y).item()


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return (y_true == y_pred).float().mean().item()


def count_parameters(model):
    """Total number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(fn, model, loss_fn, optimizer):
    """Everything you chose, as a dict for run.config (one column each in W&B)."""
    activations = [type(m).__name__ for m in model.modules()
                   if type(m).__module__ == "torch.nn.modules.activation"]
    return {
        "experiment": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "n_params": count_parameters(model),
        "n_linear_layers": sum(isinstance(m, nn.Linear) for m in model.modules()),
        "activations": activations,
        "loss": type(loss_fn).__name__,
        "optimizer": type(optimizer).__name__,
        **{f"optimizer_{k}": v for k, v in optimizer.defaults.items()
           if isinstance(v, (int, float, bool, str, tuple))},
        "model": str(model),
    }


@functools.lru_cache(maxsize=None)
def personal_entity():
    """
    Your personal W&B username, so runs go to YOUR account and not to a team
    you happen to belong to. Set the WANDB_ENTITY environment variable to
    override. Returns None (W&B's default) if it cannot be looked up, e.g.
    offline.
    """
    if os.environ.get("WANDB_ENTITY"):
        return os.environ["WANDB_ENTITY"]
    try:
        return wandb.Api().viewer.username
    except Exception:
        return None


# =============================================================================
# PART 5: Main (run every experiment, one W&B run each)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Sonar MLP experiments with wandb logging")
    p.add_argument("--experiment", type=str, default=None,
                   help="run only the experiment with this function name")
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="comp395-lab3-sonar")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    return p.parse_args()


def run_experiment(fn, args):
    """Build one experiment, train it, and log everything to its own W&B run."""
    X_train, X_test, y_train, y_test = load_data()

    torch.manual_seed(args.seed)
    model, loss_fn, optimizer = fn(X_train.shape[1])

    # Start a wandb run; config records every choice you made
    run = wandb.init(
        entity=None if args.no_wandb else personal_entity(),
        project=args.wandb_project,
        name=fn.__name__,
        group="mlp",
        config={**describe(fn, model, loss_fn, optimizer),
                "n_epochs": args.n_epochs, "batch_size": args.batch_size,
                "seed": args.seed, "dataset": "sonar",
                "n_train": len(y_train), "n_test": len(y_test)},
        mode="disabled" if args.no_wandb else None,
    )
    cfg = run.config

    try:
        print(f"\n=== {fn.__name__} ({count_parameters(model)} parameters) ===")
        losses = train(model, loss_fn, optimizer, X_train, y_train,
                       n_epochs=cfg.n_epochs, batch_size=cfg.batch_size,
                       run=run, X_val=X_test, y_val=y_test, seed=cfg.seed)

        train_acc = accuracy(y_train, predict(model, X_train))
        val_acc = accuracy(y_test, predict(model, X_test))
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        # Final results go in the run summary (shown in the runs table)
        run.summary["train_acc"] = train_acc
        run.summary["val_acc"] = val_acc
        run.summary["train_val_gap"] = train_acc - val_acc
        run.summary["final_loss"] = losses[-1]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(losses)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Average Loss')
        axes[0].set_title(f'Training Loss ({fn.__name__})')

        axes[1].bar(["train", "validation"], [train_acc, val_acc])
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Final Accuracy')

        plt.tight_layout()
        run.log({"training_results": wandb.Image(fig)})
        plt.close(fig)

        # Save the trained model as a versioned artifact
        if not args.no_wandb:
            path = f"{fn.__name__}.pt"
            torch.save(model.state_dict(), path)
            artifact = wandb.Artifact(f"sonar-{fn.__name__}", type="model")
            artifact.add_file(path)
            run.log_artifact(artifact)

        return {"n_params": count_parameters(model),
                "train_acc": train_acc, "val_acc": val_acc}
    finally:
        run.finish()


if __name__ == "__main__":
    args = parse_args()

    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [fn for fn in EXPERIMENTS if fn.__name__ == args.experiment]
        if not experiments:
            raise SystemExit(f"No experiment named {args.experiment!r} in EXPERIMENTS")

    results = {fn.__name__: run_experiment(fn, args) for fn in experiments}

    print(f"\n{'experiment':<36}{'params':>8}{'train acc':>11}{'val acc':>10}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["val_acc"], reverse=True):
        print(f"{name:<36}{r['n_params']:>8}{r['train_acc']:>11.3f}{r['val_acc']:>10.3f}")
