"""WGAN-GP training orchestrator for the MuseGAN generator/critic.

Ported from akanametov/musegan (``criterion.py`` + ``trainer.py``) into the
deepTechno layout. This is a *scaffold*: it is wired end to end (loss,
gradient penalty, critic/generator steps) but has NOT been run against real
data. See :meth:`MuseGanTrainer.sample_latent` for the 4-way latent sampler.

NEEDS-GPU-VALIDATION: wired, not run. Real training must happen on a GPU
(RunPod). No end-to-end training has been executed here — only shape tests.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
from torch import Tensor, nn

from .binary_neuron import BinaryNeuron
from .critic import MuseCritic
from .generator import MuseGenerator

logger = logging.getLogger(__name__)


class WassersteinLoss(nn.Module):
    """WGAN loss: ``-mean(pred * target)`` with target in {+1, -1}."""

    def forward(self, y_pred: Tensor, y_target: Tensor) -> Tensor:
        return -torch.mean(y_pred * y_target)


class GradientPenalty(nn.Module):
    """WGAN-GP gradient penalty on interpolated real/fake samples."""

    def forward(self, inputs: Tensor, outputs: Tensor) -> Tensor:
        grad = torch.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
        return torch.mean((1.0 - grad_norm) ** 2)


class MuseGanTrainer:
    """Orchestrate WGAN-GP training of a :class:`MuseGenerator`/:class:`MuseCritic` pair.

    Parameters
    ----------
    generator, critic:
        The models to train.
    g_optimizer, c_optimizer:
        Optimizers for the generator and critic respectively.
    device:
        Torch device string.
    gp_weight:
        Gradient-penalty coefficient (MuseGAN default 10).
    binary_neuron:
        Optional STE output layer applied to the generator output before it
        reaches the critic — turns soft pianorolls into binary ones.
    """

    def __init__(
        self,
        generator: MuseGenerator,
        critic: MuseCritic,
        g_optimizer: torch.optim.Optimizer,
        c_optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        gp_weight: float = 10.0,
        binary_neuron: Optional[BinaryNeuron] = None,
    ) -> None:
        self.device = device
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.gp_weight = gp_weight
        self.binary_neuron = binary_neuron
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)
        self.z_dimension = generator.z_dimension
        self.n_tracks = generator.n_tracks

    def sample_latent(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draw the 4-way latent (chords, style, melody, groove)."""
        chords = torch.randn(batch_size, self.z_dimension, device=self.device)
        style = torch.randn(batch_size, self.z_dimension, device=self.device)
        melody = torch.randn(batch_size, self.n_tracks, self.z_dimension, device=self.device)
        groove = torch.randn(batch_size, self.n_tracks, self.z_dimension, device=self.device)
        return chords, style, melody, groove

    def _maybe_binarize(self, fake: Tensor) -> Tensor:
        return self.binary_neuron(fake) if self.binary_neuron is not None else fake

    def _critic_step(self, real: Tensor) -> dict[str, float]:
        """One critic update on a real batch. Returns loss components."""
        batch_size = real.size(0)
        alpha = torch.rand((batch_size, 1, 1, 1, 1), device=self.device).requires_grad_()
        chords, style, melody, groove = self.sample_latent(batch_size)

        self.c_optimizer.zero_grad()
        with torch.no_grad():
            fake = self._maybe_binarize(self.generator(chords, style, melody, groove)).detach()
        realfake = alpha * real + (1.0 - alpha) * fake

        fake_pred = self.critic(fake)
        real_pred = self.critic(real)
        realfake_pred = self.critic(realfake)
        fake_loss = self.c_criterion(fake_pred, -torch.ones_like(fake_pred))
        real_loss = self.c_criterion(real_pred, torch.ones_like(real_pred))
        penalty = self.c_penalty(realfake, realfake_pred)
        closs = fake_loss + real_loss + self.gp_weight * penalty
        closs.backward(retain_graph=True)
        self.c_optimizer.step()
        return {
            "closs": closs.item(),
            "cfloss": fake_loss.item(),
            "crloss": real_loss.item(),
            "cploss": self.gp_weight * penalty.item(),
        }

    def _generator_step(self, batch_size: int) -> float:
        """One generator update. Returns the generator loss."""
        self.g_optimizer.zero_grad()
        chords, style, melody, groove = self.sample_latent(batch_size)
        fake = self._maybe_binarize(self.generator(chords, style, melody, groove))
        fake_pred = self.critic(fake)
        gloss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
        gloss.backward()
        self.g_optimizer.step()
        return gloss.item()

    def train(
        self,
        dataloader: Iterable[Tensor],
        epochs: int = 500,
        repeat: int = 5,
        display_step: int = 10,
    ) -> dict[str, list[float]]:
        """Run the WGAN-GP loop: ``repeat`` critic steps per generator step.

        NEEDS-GPU-VALIDATION: intended for GPU; not exercised in tests.
        """
        history: dict[str, list[float]] = {k: [] for k in ("gloss", "closs", "cfloss", "crloss", "cploss")}
        for epoch in range(epochs):
            epoch_loss = {k: 0.0 for k in history}
            n_batches = 0
            for real in dataloader:
                real = real.to(self.device)
                n_batches += 1
                acc = {"closs": 0.0, "cfloss": 0.0, "crloss": 0.0, "cploss": 0.0}
                for _ in range(repeat):
                    step = self._critic_step(real)
                    for k in acc:
                        acc[k] += step[k] / repeat
                for k in acc:
                    epoch_loss[k] += acc[k]
                epoch_loss["gloss"] += self._generator_step(real.size(0))
            if n_batches:
                for k in epoch_loss:
                    epoch_loss[k] /= n_batches
            for k in history:
                history[k].append(epoch_loss[k])
            if epoch % display_step == 0:
                logger.info(
                    "Epoch %d/%d | G %.3f | C %.3f (fake %.3f, real %.3f, gp %.3f)",
                    epoch, epochs, epoch_loss["gloss"], epoch_loss["closs"],
                    epoch_loss["cfloss"], epoch_loss["crloss"], epoch_loss["cploss"],
                )
        return history
