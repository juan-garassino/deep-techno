"""Guardian run #2 pass 3: MuseGAN geometry + an end-to-end trainer epoch on
tiny CPU tensors, and VisionEncoder shapes. No GPU, no data download."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepTechno.encoders.vision_encoder import VisionEncoder  # noqa: E402
from deepTechno.musegan import (  # noqa: E402
    BinaryNeuron, MuseCritic, MuseGanTrainer, MuseGenerator)

Z, TRACKS, STEPS, PITCHES = 8, 2, 16, 84


def _pair(n_bars):
    gen = MuseGenerator(z_dimension=Z, hid_channels=16, hid_features=16, n_tracks=TRACKS,
                        n_bars=n_bars, n_steps_per_bar=STEPS, n_pitches=PITCHES)
    critic = MuseCritic(hid_channels=8, hid_features=16, n_tracks=TRACKS, n_bars=n_bars,
                        n_steps_per_bar=STEPS, n_pitches=PITCHES)
    return gen, critic


@pytest.mark.parametrize("n_bars", [2, 3, 4])
def test_critic_scores_any_bar_count(n_bars):
    # The generator supports n_bars != 2 via TemporalNetwork; the critic used to
    # hard-code a 12*hid_channels flatten size for those and crash.
    gen, critic = _pair(n_bars)
    x = torch.rand(2, TRACKS, n_bars, STEPS, PITCHES)
    assert critic(x).shape == (2, 1)
    gen.eval()
    fake = gen(torch.randn(2, Z), torch.randn(2, Z),
               torch.randn(2, TRACKS, Z), torch.randn(2, TRACKS, Z))
    assert fake.shape == x.shape


def test_bar_generator_rejects_unsupported_grid():
    # The tconv chain is fixed at 16 steps x 84 pitches; fail loudly, not in view().
    with pytest.raises(ValueError):
        MuseGenerator(z_dimension=Z, hid_channels=16, hid_features=16, n_tracks=TRACKS,
                      n_steps_per_bar=32, n_pitches=PITCHES)


def test_trainer_one_epoch_updates_both_nets():
    torch.manual_seed(0)
    gen, critic = _pair(2)
    trainer = MuseGanTrainer(
        generator=gen, critic=critic,
        g_optimizer=torch.optim.Adam(gen.parameters(), lr=1e-3),
        c_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        binary_neuron=BinaryNeuron(),
    )
    g0 = [p.detach().clone() for p in gen.parameters()]
    c0 = [p.detach().clone() for p in critic.parameters()]
    data = (torch.rand(4, TRACKS, 2, STEPS, PITCHES) > 0.9).float()
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    hist = trainer.train(loader, epochs=1, repeat=2)
    assert all(len(v) == 1 and torch.isfinite(torch.tensor(v[0])) for v in hist.values())
    assert any(not torch.equal(a, b) for a, b in zip(g0, gen.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(c0, critic.parameters()))


def test_trainer_accepts_tensordataset_batches():
    # DataLoader(TensorDataset(x)) yields [x] lists; train() should unwrap them.
    gen, critic = _pair(2)
    trainer = MuseGanTrainer(gen, critic,
                             torch.optim.Adam(gen.parameters()), torch.optim.Adam(critic.parameters()))
    ds = torch.utils.data.TensorDataset(torch.rand(2, TRACKS, 2, STEPS, PITCHES))
    hist = trainer.train(torch.utils.data.DataLoader(ds, batch_size=2), epochs=1, repeat=1)
    assert len(hist["gloss"]) == 1


@pytest.mark.parametrize("size", [32, (32, 48)])
def test_vision_encoder_shapes(size):
    enc = VisionEncoder(d_model=8, img_size=size, patch_size=16)
    h, w = (size, size) if isinstance(size, int) else size
    out = enc(torch.rand(2, 3, h, w))
    assert out.shape == (2, (h // 16) * (w // 16), 8)


def test_input_validation_survives_python_O():
    # These used to be asserts (bandit B101), which `python -O` strips.
    from deepTechno.model.loss import SmoothCrossEntropyLoss
    from deepTechno.model.transformer import MusicTransformer

    with pytest.raises(ValueError):
        SmoothCrossEntropyLoss(1.5, vocab_size=5)
    model = MusicTransformer(n_layers=1, num_heads=2, d_model=8, dim_feedforward=16,
                             max_sequence=16, rpr=False).train()
    with pytest.raises(RuntimeError):
        model.generate(primer=torch.tensor([60]), target_seq_length=4)
