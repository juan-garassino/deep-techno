"""MuseGAN subsystem: multi-track WGAN-GP for symbolic techno generation.

Ported from akanametov/musegan (generator/critic/WGAN-GP) + salu133445/musegan
(binary-neuron STE concept). Feeds on multi-track pianoroll tensors produced by
:func:`deepTechno.data.preprocess.notes_to_pianoroll` from ``all_notes.csv``.

NEEDS-GPU-VALIDATION: scaffold only — shape-tested, never trained.
"""
from __future__ import annotations

from .binary_neuron import BinaryNeuron, binary_round_ste, binary_stochastic_ste
from .critic import MuseCritic
from .generator import BarGenerator, MuseGenerator, Reshape, TemporalNetwork, initialize_weights
from .training_loop import GradientPenalty, MuseGanTrainer, WassersteinLoss

__all__ = [
    "BinaryNeuron",
    "binary_round_ste",
    "binary_stochastic_ste",
    "MuseCritic",
    "BarGenerator",
    "MuseGenerator",
    "Reshape",
    "TemporalNetwork",
    "initialize_weights",
    "GradientPenalty",
    "MuseGanTrainer",
    "WassersteinLoss",
]
