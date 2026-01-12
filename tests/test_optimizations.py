"""Tests for training optimizations utilities."""

import builtins
import logging

import torch.nn as nn

from src.training.optimizations import apply_liger_kernels


class DummyModel(nn.Module):
    """Simple model used for optimization tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


def test_apply_liger_kernels_missing_dependency(monkeypatch, caplog):
    """apply_liger_kernels should warn and no-op when liger_kernel is missing."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("liger_kernel"):
            raise ImportError("liger_kernel not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    caplog.set_level(logging.WARNING, logger="src.training.optimizations")

    model = DummyModel()
    result = apply_liger_kernels(model, config=object())

    assert result is model
    assert any(
        "Liger kernels requested but `liger_kernel` is not installed" in message
        for message in caplog.messages
    )
