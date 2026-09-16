"""Multi-track MuseGAN generator with 4-way latent factorization.

Ported from akanametov/musegan (PyTorch) into the deepTechno src/ layout.

The generator factorizes noise into four latent spaces (MuseGAN, Dong et al. 2018):

- ``chords``  — shared across tracks, time-varying (via :class:`TemporalNetwork`)
- ``style``   — shared across tracks, static
- ``melody``  — per-track, time-varying (one :class:`TemporalNetwork` per track)
- ``groove``  — per-track, static

The four latents are concatenated per (bar, track) and fed to a per-track
:class:`BarGenerator`, giving a controllable-generation interface for symbolic
music. Output shape: ``(batch, n_tracks, n_bars, n_steps_per_bar, n_pitches)``.

NEEDS-GPU-VALIDATION: shape-tested on CPU only; no real training has run.
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn


class Reshape(nn.Module):
    """Reshape a tensor keeping the batch dimension fixed."""

    def __init__(self, shape: List[int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), *self.shape)


def initialize_weights(layer: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    """Normal-init conv/linear layers, zero-init biases (MuseGAN default)."""
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        nn.init.normal_(layer.weight, mean, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class TemporalNetwork(nn.Module):
    """Expand a static latent into ``n_bars`` per-bar latents via 2 tconv layers.

    z (batch, z_dim) -> (batch, z_dim, n_bars). Resizable to more bars through
    the second kernel ``(n_bars - 1, 1)``.
    """

    def __init__(self, z_dimension: int = 32, hid_channels: int = 1024, n_bars: int = 2) -> None:
        super().__init__()
        self.n_bars = n_bars
        self.net = nn.Sequential(
            # input: (batch, z_dimension)
            Reshape(shape=[z_dimension, 1, 1]),
            nn.ConvTranspose2d(z_dimension, hid_channels, kernel_size=(2, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # (batch, hid_channels, 2, 1)
            nn.ConvTranspose2d(hid_channels, z_dimension, kernel_size=(self.n_bars - 1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(z_dimension),
            nn.ReLU(inplace=True),
            # (batch, z_dimension, n_bars, 1)
            Reshape(shape=[z_dimension, self.n_bars]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BarGenerator(nn.Module):
    """Generate one bar of one track's pianoroll from a concatenated 4-way latent.

    input (batch, 4 * z_dimension) -> (batch, out_channels, 1, n_steps_per_bar, n_pitches).
    """

    def __init__(
        self,
        z_dimension: int = 32,
        hid_features: int = 1024,
        hid_channels: int = 512,
        out_channels: int = 1,
        n_steps_per_bar: int = 16,
        n_pitches: int = 84,
    ) -> None:
        super().__init__()
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        # The fixed tconv chain below hard-expands the time axis by 2*2*2 = 8 and
        # the pitch axis by 7*12 = 84, then reshapes to (n_steps_per_bar, n_pitches).
        # That reshape only works when the pre-conv feature map has 2 rows, i.e.
        # hid_features // hid_channels == 2. Fail loudly instead of a cryptic view().
        if hid_features % hid_channels != 0 or hid_features // hid_channels != 2:
            raise ValueError(
                "BarGenerator requires hid_features // hid_channels == 2 "
                f"(got hid_features={hid_features}, hid_channels={hid_channels}). "
                "Via MuseGenerator this means hid_features == hid_channels."
            )
        self.net = nn.Sequential(
            # input: (batch, 4*z_dimension)
            nn.Linear(4 * z_dimension, hid_features),
            nn.BatchNorm1d(hid_features),
            nn.ReLU(inplace=True),
            Reshape(shape=[hid_channels, hid_features // hid_channels, 1]),
            nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels, hid_channels // 2, kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels // 2, hid_channels // 2, kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            # step (time) axis expansion, then pitch axis expansion:
            nn.ConvTranspose2d(hid_channels // 2, hid_channels // 2, kernel_size=(1, 7), stride=(1, 7), padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels // 2, out_channels, kernel_size=(1, 12), stride=(1, 12), padding=0),
            # (batch, out_channels, n_steps_per_bar, n_pitches)
            Reshape(shape=[1, 1, self.n_steps_per_bar, self.n_pitches]),
            # (batch, out_channels, 1, n_steps_per_bar, n_pitches)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MuseGenerator(nn.Module):
    """Multi-track generator with 4-way latent factorization.

    Parameters
    ----------
    z_dimension:
        Per-latent noise dimension (each of chords/style/melody/groove).
    hid_channels, hid_features:
        Hidden sizes for the temporal networks and bar generators.
    out_channels:
        Output channels per track (1 for a binary pianoroll).
    n_tracks, n_bars, n_steps_per_bar, n_pitches:
        Pianoroll geometry. Techno default: 4 tracks, 2 bars, 16 steps/bar, 84 pitches.
    """

    def __init__(
        self,
        z_dimension: int = 32,
        hid_channels: int = 1024,
        hid_features: int = 1024,
        out_channels: int = 1,
        n_tracks: int = 4,
        n_bars: int = 2,
        n_steps_per_bar: int = 16,
        n_pitches: int = 84,
    ) -> None:
        super().__init__()
        self.z_dimension = z_dimension
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        # chords: shared, time-varying
        self.chords_network = TemporalNetwork(z_dimension, hid_channels, n_bars=n_bars)
        # melody: per-track, time-varying
        self.melody_networks = nn.ModuleDict(
            {
                f"melodygen_{n}": TemporalNetwork(z_dimension, hid_channels, n_bars=n_bars)
                for n in range(self.n_tracks)
            }
        )
        # one bar generator per track
        self.bar_generators = nn.ModuleDict(
            {
                f"bargen_{n}": BarGenerator(
                    z_dimension,
                    hid_features,
                    hid_channels // 2,
                    out_channels,
                    n_steps_per_bar=n_steps_per_bar,
                    n_pitches=n_pitches,
                )
                for n in range(self.n_tracks)
            }
        )

    def forward(self, chords: Tensor, style: Tensor, melody: Tensor, groove: Tensor) -> Tensor:
        """Forward.

        Shapes
        ------
        chords: (batch, z_dimension)
        style:  (batch, z_dimension)
        melody: (batch, n_tracks, z_dimension)
        groove: (batch, n_tracks, z_dimension)

        Returns
        -------
        (batch, n_tracks, n_bars, n_steps_per_bar, n_pitches)
        """
        chord_outs = self.chords_network(chords)
        bar_outs = []
        for bar in range(self.n_bars):
            track_outs = []
            chord_out = chord_outs[:, :, bar]
            style_out = style
            for track in range(self.n_tracks):
                melody_out = self.melody_networks[f"melodygen_{track}"](melody[:, track, :])[:, :, bar]
                groove_out = groove[:, track, :]
                z = torch.cat([chord_out, style_out, melody_out, groove_out], dim=1)
                track_outs.append(self.bar_generators[f"bargen_{track}"](z))
            bar_outs.append(torch.cat(track_outs, dim=1))
        return torch.cat(bar_outs, dim=2)
