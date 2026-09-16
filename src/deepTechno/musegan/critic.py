"""Multi-track WGAN critic for pianoroll tensors.

Ported from akanametov/musegan (PyTorch). The critic is a stack of 3D
convolutions over ``(n_tracks, n_bars, n_steps_per_bar, n_pitches)``; the
``(1, 1, 12)`` kernel does per-octave pitch pooling. Returns an unbounded
score per sample (WGAN, no sigmoid).

NEEDS-GPU-VALIDATION: shape-tested on CPU only; no real training has run.
"""
from __future__ import annotations

from torch import Tensor, nn


class MuseCritic(nn.Module):
    """WGAN critic over multi-track pianoroll tensors.

    Parameters
    ----------
    hid_channels, hid_features:
        Hidden sizes.
    out_features:
        Score dimension (1).
    n_tracks, n_bars, n_steps_per_bar, n_pitches:
        Pianoroll geometry — must match the generator.
    """

    def __init__(
        self,
        hid_channels: int = 128,
        hid_features: int = 1024,
        out_features: int = 1,
        n_tracks: int = 4,
        n_bars: int = 2,
        n_steps_per_bar: int = 16,
        n_pitches: int = 84,
    ) -> None:
        super().__init__()
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        # flattened feature count depends on the bar count after the temporal convs
        in_features = 4 * hid_channels if n_bars == 2 else 12 * hid_channels
        self.net = nn.Sequential(
            # input: (batch, n_tracks, n_bars, n_steps_per_bar, n_pitches)
            nn.Conv3d(self.n_tracks, hid_channels, (2, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, (self.n_bars - 1, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # per-octave pitch pooling:
            nn.Conv3d(hid_channels, hid_channels, (1, 1, 12), (1, 1, 12), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, (1, 1, 7), (1, 1, 7), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # time-axis downsampling:
            nn.Conv3d(hid_channels, hid_channels, (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, 2 * hid_channels, (1, 4, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(2 * hid_channels, 4 * hid_channels, (1, 3, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features, hid_features),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(hid_features, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
