"""
Automated Tests for the Sonar MLP Lab
COMP 395 – Deep Learning

DO NOT MODIFY THIS FILE.

These tests never talk to Weights & Biases: WANDB_MODE is set to "disabled"
below and train() is called without a run. You do not need to be logged in
to W&B for them to pass. Training here uses the default settings of
`python sonar_mlp.py` (100 epochs, batch size 1, seed 0).

Run with: python -m pytest test_sonar_mlp.py -v
"""

import copy
import math
import os

os.environ["WANDB_MODE"] = "disabled"

import pytest
import torch
import torch.nn as nn

from sonar_data import load_data
from sonar_mlp import EXPERIMENTS, train, predict, accuracy

N_FEATURES = 60
MIN_EXPERIMENTS = 5
LINEAR_BASELINE_VAL_ACC = 0.81    # what linear_baseline_wandb.py gets
TARGET_VAL_ACC = 0.85             # your best experiment must reach this


# =============================================================================
# Helper Functions
# =============================================================================

def _build(fn):
    """Call an experiment function with a fixed seed and check what it returns."""
    torch.manual_seed(0)
    result = fn(N_FEATURES)
    assert isinstance(result, tuple) and len(result) == 3, (
        "must return (model, loss_fn, optimizer)"
    )
    for name, value in zip(("model", "loss_fn", "optimizer"), result):
        assert value is not None, f"{name} is still TODO"
    return result


def _check_all(check):
    """Run check(fn, model, loss_fn, optimizer) on every experiment; report every failure at once."""
    failures = []
    for fn in EXPERIMENTS:
        try:
            check(fn, *_build(fn))
        except AssertionError as e:
            failures.append(f"{fn.__name__}: {str(e).splitlines()[0]}")
        except Exception as e:
            failures.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    if failures:
        pytest.fail("\n".join(failures), pytrace=False)


@torch.no_grad()
def _probabilities(model, x):
    model.eval()
    return model(x).reshape(-1)


def _is_activation(module):
    return type(module).__module__ == "torch.nn.modules.activation"


def _hidden_activations(model):
    """Activation classes used in hidden layers (the final output sigmoid is excluded)."""
    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    if leaves and isinstance(leaves[-1], nn.Sigmoid):
        leaves = leaves[:-1]
    return {type(m).__name__ for m in leaves if _is_activation(m)}


# =============================================================================
# Test: Experiment Definitions
# =============================================================================

class TestExperiments:
    """Every experiment returns a valid model, loss function, and optimizer."""

    def test_at_least_five_experiments(self):
        """EXPERIMENTS should contain at least 5 experiment functions"""
        assert len(EXPERIMENTS) >= MIN_EXPERIMENTS, (
            f"EXPERIMENTS has {len(EXPERIMENTS)} entries, expected at least {MIN_EXPERIMENTS}"
        )

    def test_returns_model_loss_optimizer(self):
        """Each experiment should return (model, loss_fn, optimizer), none of them TODO"""
        _check_all(lambda fn, model, loss_fn, optimizer: None)

    def test_model_uses_sequential(self):
        """Each model should be built with nn.Sequential"""
        def check(fn, model, loss_fn, optimizer):
            assert isinstance(model, nn.Module), "model is not an nn.Module"
            assert any(isinstance(m, nn.Sequential) for m in model.modules()), (
                "model should be built with nn.Sequential (read its documentation)"
            )
        _check_all(check)

    def test_loss_function(self):
        """loss_fn should be a torch.nn loss that works on probabilities"""
        def check(fn, model, loss_fn, optimizer):
            assert isinstance(loss_fn, nn.Module), (
                "loss_fn should be a loss object from torch.nn (e.g. created with nn.SomeLoss())"
            )
            assert not isinstance(loss_fn, nn.BCEWithLogitsLoss), (
                "BCEWithLogitsLoss expects logits, but your model already ends in a "
                "sigmoid and outputs probabilities. Read the docs: which BCE loss fits?"
            )
            p = torch.full((4,), 0.5)
            y = torch.tensor([0.0, 1.0, 0.0, 1.0])
            value = loss_fn(p, y)
            assert value.dim() == 0 and torch.isfinite(value), (
                "loss_fn(predictions, labels) should return one finite number"
            )
        _check_all(check)

    def test_optimizer_bound_to_model(self):
        """optimizer should be a torch.optim optimizer over exactly this model's parameters"""
        def check(fn, model, loss_fn, optimizer):
            assert isinstance(optimizer, torch.optim.Optimizer), (
                "optimizer should be an optimizer object from torch.optim"
            )
            opt_params = {id(p) for g in optimizer.param_groups for p in g["params"]}
            model_params = {id(p) for p in model.parameters()}
            assert opt_params == model_params, (
                "optimizer is not updating this model's parameters. "
                "Build it with model.parameters() of the model you return."
            )
        _check_all(check)

    def test_output_shape(self):
        """A (8, 60) batch should give 8 predictions, shape (8,) or (8, 1)"""
        def check(fn, model, loss_fn, optimizer):
            out = model.eval()(torch.randn(8, N_FEATURES))
            assert tuple(out.shape) in [(8,), (8, 1)], (
                f"output shape {tuple(out.shape)}, expected (8,) or (8, 1)"
            )
        _check_all(check)

    def test_output_is_probability(self):
        """Outputs should be probabilities in [0, 1], even for large inputs (end with a sigmoid)"""
        def check(fn, model, loss_fn, optimizer):
            p = _probabilities(model, 10 * torch.randn(64, N_FEATURES))
            assert torch.isfinite(p).all(), "output contains NaN or inf"
            assert (p >= 0).all() and (p <= 1).all(), (
                f"outputs range from {p.min().item():.3f} to {p.max().item():.3f}; "
                "they must be probabilities in [0, 1]. Did you forget the final sigmoid?"
            )
        _check_all(check)

    def test_has_hidden_layer(self):
        """Each model should have at least one hidden layer (2+ weight matrices)"""
        def check(fn, model, loss_fn, optimizer):
            n_weight_matrices = sum(1 for p in model.parameters() if p.dim() >= 2)
            assert n_weight_matrices >= 2, (
                f"found {n_weight_matrices} weight matrices; an MLP needs at least 2 "
                "(input -> hidden, hidden -> output). The linear baseline is already "
                "in linear_baseline_wandb.py."
            )
        _check_all(check)

    def test_docstrings_filled_in(self):
        """Each experiment should have a docstring describing it (not the TODO placeholder)"""
        def check(fn, model, loss_fn, optimizer):
            doc = (fn.__doc__ or "").strip()
            assert doc and not doc.upper().startswith("TODO"), (
                "still has the TODO docstring; describe the experiment and what it tests"
            )
        _check_all(check)

    def test_experiments_are_distinct(self):
        """No two experiments may have the same model, loss, AND optimizer settings"""
        seen = {}
        duplicates = []
        for fn in EXPERIMENTS:
            model, loss_fn, optimizer = _build(fn)
            settings = tuple(sorted((k, repr(v)) for k, v in optimizer.defaults.items()))
            key = (str(model), repr(loss_fn), type(optimizer).__name__, settings)
            if key in seen:
                duplicates.append(f"{fn.__name__} is identical to {seen[key]}")
            seen[key] = fn.__name__
        if duplicates:
            pytest.fail("\n".join(duplicates), pytrace=False)


# =============================================================================
# Test: Your Choices
# =============================================================================

class TestChoices:
    """You explored the documentation: more than one activation and optimizer."""

    def test_two_hidden_activations(self):
        """Across experiments, use at least 2 different hidden-layer activation functions"""
        used = set()
        for fn in EXPERIMENTS:
            used |= _hidden_activations(_build(fn)[0])
        assert len(used) >= 2, (
            f"hidden layers only use {sorted(used) or 'no activations'}; try at least one "
            "other activation from the torch.nn 'Non-linear Activations' docs"
        )

    def test_two_optimizers(self):
        """Across experiments, use at least 2 different torch.optim optimizers"""
        used = {type(_build(fn)[2]).__name__ for fn in EXPERIMENTS}
        assert len(used) >= 2, (
            f"every experiment uses {sorted(used)}; try at least one other optimizer "
            "from the torch.optim docs"
        )


# =============================================================================
# Test: Nonlinearity
# =============================================================================

class TestNonlinearity:
    """Stacked linear layers with no activation in between are still linear."""

    def test_not_secretly_linear(self):
        """Each model's logit should NOT be a linear function of its input"""
        # If the network is Linear -> Linear -> ... -> Sigmoid with nothing in
        # between, then logit(y_hat) = w . x + b for some w, b. A linear
        # function f satisfies f((a + b) / 2) = (f(a) + f(b)) / 2 for every
        # pair of points; we look for a pair where that fails.
        def check(fn, model, loss_fn, optimizer):
            model = copy.deepcopy(model).double().eval()
            gen = torch.Generator().manual_seed(0)
            a = torch.randn(256, N_FEATURES, generator=gen, dtype=torch.float64)
            b = torch.randn(256, N_FEATURES, generator=gen, dtype=torch.float64)

            def logit(x):
                p = _probabilities(model, x).clamp(1e-12, 1 - 1e-12)
                return torch.log(p) - torch.log1p(-p)

            gap = (logit((a + b) / 2) - (logit(a) + logit(b)) / 2).abs().max().item()
            assert gap > 1e-6, (
                "behaves like a LINEAR model (logit is linear in x). Put a nonlinear "
                "activation between your nn.Linear layers."
            )
        _check_all(check)


# =============================================================================
# Test: Training
# =============================================================================

@pytest.fixture(scope="module")
def trained():
    """Train every experiment once (no W&B) and share the results."""
    X_train, X_test, y_train, y_test = load_data()
    results = {}
    for fn in EXPERIMENTS:
        model, loss_fn, optimizer = _build(fn)
        losses = train(model, loss_fn, optimizer, X_train, y_train,
                       n_epochs=100, batch_size=1, verbose=False, seed=0)
        results[fn.__name__] = {
            "losses": losses,
            "val_acc": accuracy(y_test, predict(model, X_test)),
        }
    return results


class TestTraining:
    """Every experiment trains end to end through the provided train()."""

    def test_every_experiment_trains(self, trained):
        """train() should finish every experiment with finite losses"""
        failures = [f"{name}: loss became NaN or inf (try a smaller learning rate)"
                    for name, r in trained.items()
                    if not all(math.isfinite(v) for v in r["losses"])]
        if failures:
            pytest.fail("\n".join(failures), pytrace=False)

    def test_one_loss_per_epoch(self, trained):
        """train() should record one loss per epoch"""
        for name, r in trained.items():
            assert len(r["losses"]) == 100, (
                f"{name}: expected 100 epochs of losses, got {len(r['losses'])}"
            )


class TestPerformance:
    """Your best experiment should beat the linear baseline."""

    def test_best_experiment_beats_linear_baseline(self, trained):
        """Best final validation accuracy should be at least 0.85 (linear baseline: ~0.81)"""
        best_name, best = max(trained.items(), key=lambda kv: kv[1]["val_acc"])
        assert best["val_acc"] >= TARGET_VAL_ACC, (
            f"best experiment is {best_name} with validation accuracy {best['val_acc']:.3f}; "
            f"expected >= {TARGET_VAL_ACC} (the linear baseline gets ~{LINEAR_BASELINE_VAL_ACC}). "
            "Check your W&B curves: is the loss going down?"
        )


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
