"""Guardian run #2 regression tests: decode of special tokens, beam-search
indexing, empty-MIDI parsing and the label-smoothed loss default."""
from __future__ import annotations

import pretty_midi
import pytest

from deepTechno.data.preprocess import midi_to_notes
from deepTechno.data.tokenizer import decode_midi
from deepTechno.model.constants import TOKEN_END, TOKEN_PAD


def test_decode_midi_ignores_end_and_pad_tokens(tmp_path):
    # generate_from_dataset feeds a padded primer (..., END, PAD, PAD) into
    # generation; decoding must not turn END/PAD into velocity 128/132 events.
    on, off, shift = 60, 128 + 60, 256 + 49
    vel = 128 + 128 + 100 + 25  # velocity token -> velocity 100
    # A PAD after the velocity token used to override it with velocity 132.
    seq = [vel, TOKEN_PAD, on, shift, off, TOKEN_END, TOKEN_PAD, TOKEN_PAD]
    out = tmp_path / "out.mid"
    mid = decode_midi(seq, file_path=str(out))
    notes = mid.instruments[0].notes
    assert [n.pitch for n in notes] == [60]
    assert [n.velocity for n in notes] == [100]
    assert len(pretty_midi.PrettyMIDI(str(out)).instruments[0].notes) == 1


def test_decode_midi_stops_at_end_token():
    on, off = 60, 128 + 60
    shift = 256 + 49  # 0.5 s
    seq = [on, shift, off, TOKEN_END, on + 2, shift, off + 2]
    notes = decode_midi(seq).instruments[0].notes
    assert [n.pitch for n in notes] == [60]


def test_midi_to_notes_empty_midi_returns_empty_frame(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    path = tmp_path / "empty.mid"
    pm.write(str(path))
    df = midi_to_notes(str(path))
    assert len(df) == 0
    assert list(df.columns) == ["pitch", "start", "end", "step", "duration"]

    pm.instruments.append(pretty_midi.Instrument(program=0))
    pm.write(str(path))
    assert len(midi_to_notes(str(path))) == 0


def test_smooth_ce_default_ignore_index():
    torch = pytest.importorskip("torch")
    from deepTechno.model.loss import SmoothCrossEntropyLoss

    loss = SmoothCrossEntropyLoss(0.1, vocab_size=5)  # ignore_index=-100
    logits = torch.randn(4, 5)
    target = torch.tensor([1, -100, 3, 0])
    val = loss(logits, target)
    assert torch.isfinite(val)
    # The ignored position must not contribute.
    ref = SmoothCrossEntropyLoss(0.1, vocab_size=5)(logits[[0, 2, 3]], target[[0, 2, 3]])
    assert torch.allclose(val, ref)


def test_beam_search_picks_correct_rows_and_tokens(monkeypatch):
    torch = pytest.importorskip("torch")
    from deepTechno.model.transformer import MusicTransformer

    model = MusicTransformer(n_layers=1, num_heads=2, d_model=8, dim_feedforward=16,
                             max_sequence=16, rpr=False).eval()
    favoured = 300  # a time-shift token, well inside [0, TOKEN_END)

    def fake_forward(x, mask=True, encoder_memory=None):
        logits = torch.full((x.shape[0], x.shape[1], TOKEN_PAD + 1), -1e4)
        logits[..., favoured] = 10.0
        logits[..., favoured - 1] = 9.0
        return logits

    monkeypatch.setattr(model, "forward", fake_forward)
    primer = torch.tensor([60, 300], dtype=torch.long)
    out = model.generate(primer=primer, target_seq_length=6, beam=2)
    generated = out[:, 2:]
    # Every generated token must come from the favoured pair, never END/PAD or
    # a token shifted by the wrong row width.
    assert set(generated.flatten().tolist()) <= {favoured, favoured - 1}
    assert out.shape == (2, 6)
