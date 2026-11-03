# A Random Forest combines decision trees trained on bootstrap samples.
# Each split considers a random subset of features, making the trees less similar.
# Averaging their class probabilities reduces variance compared with one tree.
# On this small Sonar dataset, I expect it to beat the linear baseline and be
# competitive with the MLPs because it can model nonlinear feature interactions.

import os

import wandb
from sklearn.ensemble import RandomForestClassifier

from sonar_data import load_data
from sonar_mlp import personal_entity


def main():
    X_train, X_val, y_train, y_val = load_data()
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled"
    run = wandb.init(
        entity=None if disabled else personal_entity(),
        project="comp395-lab3-sonar",
        name="random_forest",
        group="baseline",
        config={
            "model": "RandomForestClassifier",
            **model.get_params(),
            "dataset": "sonar",
            "n_train": len(y_train),
            "n_test": len(y_val),
        },
    )
    try:
        model.fit(X_train.numpy(), y_train.numpy())
        train_acc = model.score(X_train.numpy(), y_train.numpy())
        val_acc = model.score(X_val.numpy(), y_val.numpy())
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        run.summary["train_acc"] = train_acc
        run.summary["val_acc"] = val_acc
        run.summary["train_val_gap"] = train_acc - val_acc
    finally:
        run.finish()


if __name__ == "__main__":
    main()

# With 200 trees and random_state=0, the forest scored 100% on training data
# and 85.71% on the shared validation split, compared with 92.86% for the best MLP.
# It beat the linear baseline (80.95%), but the MLP classified three more of
# the 42 validation examples correctly; this one split does not establish a general winner.
