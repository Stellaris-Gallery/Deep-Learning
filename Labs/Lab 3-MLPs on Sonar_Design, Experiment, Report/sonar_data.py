"""
Sonar Data Loading (provided)
COMP 395 – Deep Learning

DO NOT MODIFY THIS FILE. Every script in this lab (and the autograder) uses
load_data() so that everyone trains and tests on exactly the same split.

Dataset: UCI "Connectionist Bench (Sonar, Mines vs. Rocks)"
  208 sonar returns, 60 features each (energy in a frequency band, 0 to 1)
  label 1 = Mine (metal cylinder), label 0 = Rock
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "sonar.csv"


def load_data():
    """
    Load and preprocess the Sonar dataset.

    Returns:
        X_train: (166, 60) float32 tensor, normalized
        X_test:  (42, 60)  float32 tensor, normalized with TRAINING statistics
        y_train: (166,)    float32 tensor of 0/1 labels
        y_test:  (42,)     float32 tensor of 0/1 labels
    """
    raw = np.loadtxt(DATA_PATH, delimiter=",", dtype=str)
    X = raw[:, :-1].astype(np.float32)
    y = (raw[:, -1] == "M").astype(np.float32)   # Mine = 1, Rock = 0

    # stratify keeps the mine/rock ratio the same in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # Normalize using training statistics only (no peeking at the test set)
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training samples: {len(y_train)}  (mines: {int(y_train.sum())})")
    print(f"Test samples:     {len(y_test)}  (mines: {int(y_test.sum())})")
    print(f"Features:         {X_train.shape[1]}")
