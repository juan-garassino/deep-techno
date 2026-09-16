"""Binary-neuron straight-through estimator (STE) output layer.

Concept ported from salu133445/musegan (``binary_ops.py``). The original TF1
``gradient_override_map`` code is unportable; this is a modern PyTorch
reimplementation via ``torch.autograd.Function`` so the generator can emit
genuinely binary pianorolls (deterministic threshold or Bernoulli sample)
while gradients pass through as the identity (straight-through).

NEEDS-GPU-VALIDATION: shape-tested on CPU only; no real training has run.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class _RoundSTE(torch.autograd.Function):
    """Round to {0, 1} on the forward pass, identity gradient on the backward pass."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # noqa: D401
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # noqa: D401
        return grad_output


class _BernoulliSTE(torch.autograd.Function):
    """Bernoulli-sample in the forward pass, identity gradient on the backward pass."""

    @staticmethod
    def forward(ctx, probs: Tensor) -> Tensor:  # noqa: D401
        return torch.bernoulli(probs)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # noqa: D401
        return grad_output


def binary_round_ste(x: Tensor) -> Tensor:
    """Deterministic binary threshold at 0.5 with straight-through gradient."""
    return _RoundSTE.apply(x)


def binary_stochastic_ste(probs: Tensor) -> Tensor:
    """Stochastic (Bernoulli) binarization with straight-through gradient."""
    return _BernoulliSTE.apply(probs)


class BinaryNeuron(nn.Module):
    """Sigmoid -> binary STE output layer for a generator.

    Parameters
    ----------
    stochastic:
        If ``True`` use a Bernoulli sample (stochastic STE); otherwise
        deterministic rounding at 0.5.
    slope:
        Sigmoid slope (annealing schedule hook, salu133445's ``nodes['slope']``);
        raise it over training to sharpen the sigmoid.
    """

    def __init__(self, stochastic: bool = False, slope: float = 1.0) -> None:
        super().__init__()
        self.stochastic = stochastic
        self.slope = slope

    def forward(self, logits: Tensor) -> Tensor:
        probs = torch.sigmoid(self.slope * logits)
        if self.stochastic:
            return binary_stochastic_ste(probs)
        return binary_round_ste(probs)
