"""Shape / import tests for the MuseGAN subsystem and the CSV->pianoroll bridge.

These are shape-only smoke tests: they never train. They verify that the
generator produces the expected pianoroll shape from a random 4-way latent,
that the critic scores that pianoroll, and that the preprocess bridge converts
a tiny CSV slice into the right tensor shape.

NEEDS-GPU-VALIDATION: real training is untested (GPU/RunPod only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from deepTechno.data.preprocess import (
    csv_to_pianoroll_dataset,
    notes_to_pianoroll,
)
from deepTechno.musegan import (
    BinaryNeuron,
    MuseCritic,
    MuseGanTrainer,
    MuseGenerator,
)

# Small geometry so the tests run fast on CPU. n_bars=2 keeps the critic's
# flatten dimension on its documented 4*hid_channels path.
#
# BarGenerator's fixed tconv chain requires (hid_features // bar_hid_channels)==2,
# where the generator passes bar_hid_channels = GEN_HID_CHANNELS // 2. So we must
# keep GEN_HID_FEATURES == GEN_HID_CHANNELS.
Z = 16
N_TRACKS = 4
N_BARS = 2
N_STEPS = 16
N_PITCHES = 84
GEN_HID_CHANNELS = 64
GEN_HID_FEATURES = 64  # == GEN_HID_CHANNELS to satisfy the ratio invariant
CRITIC_HID_C = 32
CRITIC_HID_F = 128


def _make_generator() -> MuseGenerator:
    return MuseGenerator(
        z_dimension=Z, hid_channels=GEN_HID_CHANNELS, hid_features=GEN_HID_FEATURES,
        n_tracks=N_TRACKS, n_bars=N_BARS, n_steps_per_bar=N_STEPS, n_pitches=N_PITCHES,
    )


def _make_critic() -> MuseCritic:
    return MuseCritic(
        hid_channels=CRITIC_HID_C, hid_features=CRITIC_HID_F, out_features=1,
        n_tracks=N_TRACKS, n_bars=N_BARS, n_steps_per_bar=N_STEPS, n_pitches=N_PITCHES,
    )


def _latent(batch: int):
    return (
        torch.randn(batch, Z),
        torch.randn(batch, Z),
        torch.randn(batch, N_TRACKS, Z),
        torch.randn(batch, N_TRACKS, Z),
    )


def test_generator_forward_shape():
    gen = _make_generator().eval()
    batch = 2
    with torch.no_grad():
        out = gen(*_latent(batch))
    assert tuple(out.shape) == (batch, N_TRACKS, N_BARS, N_STEPS, N_PITCHES)


def test_critic_scores_generated_pianoroll():
    gen = _make_generator().eval()
    critic = _make_critic().eval()
    batch = 2
    with torch.no_grad():
        fake = gen(*_latent(batch))
        score = critic(fake)
    assert tuple(score.shape) == (batch, 1)


def test_binary_neuron_produces_binary_output_and_shape():
    gen = _make_generator().eval()
    binary = BinaryNeuron(stochastic=False)
    batch = 2
    with torch.no_grad():
        fake = gen(*_latent(batch))
        binarized = binary(fake)
    assert binarized.shape == fake.shape
    unique = torch.unique(binarized)
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_trainer_wires_and_takes_one_critic_step():
    gen = _make_generator()
    critic = _make_critic()
    trainer = MuseGanTrainer(
        generator=gen,
        critic=critic,
        g_optimizer=torch.optim.Adam(gen.parameters(), lr=1e-4),
        c_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-4),
        device="cpu",
    )
    chords, style, melody, groove = trainer.sample_latent(2)
    assert chords.shape == (2, Z)
    assert melody.shape == (2, N_TRACKS, Z)
    # A real batch is needed for a critic step; batchnorm needs batch >= 2.
    real = torch.rand(2, N_TRACKS, N_BARS, N_STEPS, N_PITCHES)
    losses = trainer._critic_step(real)
    assert set(losses) == {"closs", "cfloss", "crloss", "cploss"}
    assert all(np.isfinite(v) for v in losses.values())


def test_notes_to_pianoroll_shape():
    notes = pd.DataFrame(
        {
            "pitch": [36, 60, 72, 48],
            "start": [0.0, 0.117, 0.234, 0.469],
            "end": [0.05, 0.16, 0.28, 0.5],
            "step": [0.0, 0.117, 0.117, 0.235],
            "duration": [0.05, 0.05, 0.05, 0.05],
        }
    )
    roll = notes_to_pianoroll(
        notes, n_tracks=N_TRACKS, n_bars=N_BARS,
        n_steps_per_bar=N_STEPS, n_pitches=N_PITCHES, bpm=128.0,
    )
    assert roll.shape == (N_TRACKS, N_BARS, N_STEPS, N_PITCHES)
    assert roll.dtype == np.float32
    assert roll.sum() > 0  # at least some notes landed on the grid
    assert set(np.unique(roll).tolist()).issubset({0.0, 1.0})


def test_csv_to_pianoroll_dataset_shape(tmp_path):
    csv = tmp_path / "tiny_notes.csv"
    pd.DataFrame(
        {
            "pitch": [36, 40, 60, 72, 48, 55],
            "start": [0.0, 0.1, 0.2, 1.0, 1.1, 2.0],
            "end": [0.05, 0.15, 0.25, 1.05, 1.15, 2.05],
            "step": [0.0, 0.1, 0.1, 0.8, 0.1, 0.9],
            "duration": [0.05] * 6,
        }
    ).to_csv(csv, index=False)
    ds = csv_to_pianoroll_dataset(
        str(csv), n_tracks=N_TRACKS, n_bars=N_BARS,
        n_steps_per_bar=N_STEPS, n_pitches=N_PITCHES, bpm=128.0,
    )
    assert ds.ndim == 5
    assert ds.shape[1:] == (N_TRACKS, N_BARS, N_STEPS, N_PITCHES)
    assert ds.shape[0] >= 1


def test_pianoroll_matches_generator_output_geometry():
    """The bridge tensor must be feedable to the critic that scores the generator."""
    gen = _make_generator().eval()
    critic = _make_critic().eval()
    notes = pd.DataFrame({"pitch": [36, 60, 72], "start": [0.0, 0.5, 1.0],
                          "end": [0.1, 0.6, 1.1], "step": [0.0, 0.5, 0.5],
                          "duration": [0.1, 0.1, 0.1]})
    roll = notes_to_pianoroll(notes, n_tracks=N_TRACKS, n_bars=N_BARS,
                              n_steps_per_bar=N_STEPS, n_pitches=N_PITCHES)
    real = torch.from_numpy(np.stack([roll, roll], axis=0))  # batch of 2
    with torch.no_grad():
        gen_out = gen(*_latent(2))
        assert real.shape == gen_out.shape
        assert critic(real).shape == (2, 1)
