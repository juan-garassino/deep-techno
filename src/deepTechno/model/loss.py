import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SmoothCrossEntropyLoss(_Loss):
    """Label-smoothed cross entropy (Szegedy et al. 2016, https://arxiv.org/abs/1512.00567)."""

    __constants__ = ["label_smoothing", "vocab_size", "ignore_index", "reduction"]

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction="mean", is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input, target):
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)
        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == "mean":
            return ce.sum() / torch.sum(target != self.ignore_index)
        elif self.reduction == "sum":
            return ce.sum()
        raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)


def mse_with_positive_pressure(y_true, y_pred):
    """MSE loss penalising negative predictions (used for step/duration heads in LSTM)."""
    import tensorflow as tf
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)
