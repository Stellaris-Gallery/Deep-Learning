"""
Linear Baseline on Sonar, with Weights & Biases Tracking
COMP 395 – Deep Learning

THIS FILE IS COMPLETE. You do not need to edit it, and it is not graded.
Read it top to bottom, run it, and then look at your W&B dashboard.

The model is the one you built in Lab 2 (one sigmoid neuron, squared-error
loss, SGD with one sample per update, weights start at zero), now built with
nn.Sequential and trained on the Sonar data. Two things are new:

  * PART 1 is an "experiment function" that builds the model, the loss, and
    the optimizer. Your experiments in sonar_mlp.py have the same shape.
  * Weights & Biases tracking: look for `run = wandb.init(...)`,
    `run.log(...)`, `run.summary[...]`, and `run.finish()`.

The structure (Parts 1-5, parse_args, train(..., run=None), the try/finally
around the run) is the same as binary_classification_wandb.py, the W&B
version of Lab 2 you have already seen. Compare the two side by side.

Run `python linear_baseline_wandb.py --help` for training options.
"""

import argparse
import os

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # save figures without opening windows
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import TensorDataset, DataLoader

from sonar_data import load_data


# =============================================================================
# PART 1: The Experiment (model, loss, optimizer)
# =============================================================================

def linear_baseline(n_features):
    """One sigmoid neuron (Lab 2's model): how far does a linear model get on Sonar?"""
    model = nn.Sequential(
        nn.Linear(n_features, 1),
        nn.Sigmoid(),
    )
    # Same as Lab 2: start from w = 0, b = 0
    nn.init.zeros_(model[0].weight)
    nn.init.zeros_(model[0].bias)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, loss_fn, optimizer


# =============================================================================
# PART 2: Data Loading (provided, see sonar_data.py)
# =============================================================================


# =============================================================================
# PART 3: Training Loop
# =============================================================================

def train(model, loss_fn, optimizer, X_train, y_train, n_epochs=100,
          batch_size=1, verbose=True, run=None, X_val=None, y_val=None, seed=0):
    """
    Train the model with the given loss and optimizer.

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
            optimizer.zero_grad()   # gradients accumulate otherwise
            loss.backward()         # computes dL/dw and dL/db
            optimizer.step()        # w -= lr * dw ; b -= lr * db

            epoch_loss += loss.item() * len(y_batch)

        avg_loss = epoch_loss / len(y_train)
        losses.append(avg_loss)

        # Log this epoch to wandb (skipped when no run is passed)
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
# PART 4: Evaluation
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


def personal_entity():
    """Your W&B username, so runs land in YOUR account, not a team you're in."""
    if os.environ.get("WANDB_ENTITY"):
        return os.environ["WANDB_ENTITY"]
    try:
        return wandb.Api().viewer.username
    except Exception:
        return None   # offline or not logged in: let W&B use its default


# =============================================================================
# PART 5: Main (run training and evaluation)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Linear baseline on Sonar with wandb logging")
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="comp395-lab3-sonar")
    p.add_argument("--wandb_run_name", type=str, default="linear_baseline")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}")

    torch.manual_seed(args.seed)
    model, loss_fn, optimizer = linear_baseline(X_train.shape[1])

    # Start a wandb run; config records the choices that define this run
    run = wandb.init(
        entity=None if args.no_wandb else personal_entity(),   # your account
        project=args.wandb_project,
        name=args.wandb_run_name,
        group="baseline",
        config={"experiment": "linear_baseline",
                "n_params": sum(p.numel() for p in model.parameters()),
                "loss": type(loss_fn).__name__,
                "optimizer": type(optimizer).__name__,
                "optimizer_lr": optimizer.defaults["lr"],
                "n_epochs": args.n_epochs, "batch_size": args.batch_size,
                "seed": args.seed, "dataset": "sonar",
                "n_train": len(y_train), "n_test": len(y_test)},
        mode="disabled" if args.no_wandb else None,
    )
    cfg = run.config

    try:
        # Train (per-epoch metrics are logged inside train)
        print("\nTraining...")
        losses = train(model, loss_fn, optimizer, X_train, y_train,
                       n_epochs=cfg.n_epochs, batch_size=cfg.batch_size,
                       run=run, X_val=X_test, y_val=y_test, seed=cfg.seed)

        # Evaluate
        print("\nEvaluating...")
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
        w = model[0].weight.detach().squeeze(0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(losses)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Average Loss')
        axes[0].set_title('Training Loss (linear baseline)')

        axes[1].bar(range(len(w)), w.numpy())
        axes[1].set_xlabel('Feature Index')
        axes[1].set_ylabel('Weight')
        axes[1].set_title('Learned Weights')

        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150)
        print("\nPlot saved to training_results.png")

        # Log the figure and a per-feature weight table
        run.log({"training_results": wandb.Image(fig)})
        weight_table = wandb.Table(
            columns=["feature", "weight"],
            data=[[i, wt] for i, wt in enumerate(w.tolist())],
        )
        run.log({"learned_weights": weight_table})

        # Save the learned parameters as a versioned model artifact
        torch.save(model.state_dict(), "linear_baseline.pt")
        if not args.no_wandb:
            artifact = wandb.Artifact("sonar-linear_baseline", type="model")
            artifact.add_file("linear_baseline.pt")
            run.log_artifact(artifact)
    finally:
        run.finish()
