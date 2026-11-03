"""
Automated Tests for the Random Forest Comparison (Step 4)
COMP 395 – Deep Learning

DO NOT MODIFY THIS FILE.

comparison.py is run with WANDB_MODE=disabled, so these tests pass without a
W&B login. (Your real run still has to show up in your W&B project for the
report.)

Run with: python -m pytest test_comparison.py -v
"""

import ast
import os
import subprocess
import sys

import pytest


# =============================================================================
# Helper: static analysis of comparison.py
# =============================================================================

def _parse_source():
    """Read and parse comparison.py source code."""
    with open("comparison.py", "r") as f:
        source = f.read()
    tree = ast.parse(source)
    return source, tree


def _get_imported_modules(tree):
    """Extract all imported module names from the AST."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _require_started(tree):
    """An untouched comparison.py earns no credit: it must at least import sklearn."""
    assert "sklearn" in _get_imported_modules(tree), (
        "comparison.py has not been started yet "
        "(it should import RandomForestClassifier from sklearn)"
    )


# =============================================================================
# Tests
# =============================================================================

class TestComparison:
    """Tests for the Random Forest comparison file."""

    def test_has_description(self):
        """comparison.py should contain a descriptive comment or docstring (>50 chars)"""
        source, tree = _parse_source()
        _require_started(tree)
        comment_chars = sum(
            len(line.strip()) - 1
            for line in source.splitlines()
            if line.strip().startswith("#") and "TODO" not in line
        )
        docstring = ast.get_docstring(tree) or ""
        total_description = comment_chars + len(docstring)
        assert total_description >= 50, (
            "comparison.py should include a descriptive comment or docstring "
            f"(at least 50 characters; found {total_description})"
        )

    def test_uses_random_forest(self):
        """comparison.py should use sklearn's RandomForestClassifier"""
        source, tree = _parse_source()
        _require_started(tree)
        assert "RandomForestClassifier" in source, (
            "comparison.py should train a RandomForestClassifier"
        )

    def test_uses_sonar_data(self):
        """comparison.py should use load_data() from sonar_data.py (same split as the MLPs)"""
        source, tree = _parse_source()
        _require_started(tree)
        assert "sonar_data" in _get_imported_modules(tree) and "load_data" in source, (
            "comparison.py should import load_data from sonar_data so the forest "
            "is tested on exactly the same split as your MLPs"
        )

    def test_logs_to_wandb(self):
        """comparison.py should log the forest to W&B (import wandb, wandb.init, run.finish)"""
        source, tree = _parse_source()
        _require_started(tree)
        assert "wandb" in _get_imported_modules(tree), "comparison.py should import wandb"
        assert "wandb.init" in source, "comparison.py should start a run with wandb.init(...)"
        assert ".finish(" in source or "with wandb.init" in source, (
            "comparison.py should close its run with run.finish()"
        )

    def test_runs_successfully(self):
        """comparison.py should run without errors"""
        _, tree = _parse_source()
        _require_started(tree)
        env = {**os.environ, "WANDB_MODE": "disabled"}
        result = subprocess.run(
            [sys.executable, "comparison.py"],
            capture_output=True, text=True, timeout=120, env=env
        )
        assert result.returncode == 0, (
            f"comparison.py failed with error:\n{result.stderr}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
