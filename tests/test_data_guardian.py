"""Regression tests for the tokenizer and the all_notes.csv -> pianoroll bridge.

Pure pretty_midi / numpy / pandas: no torch, no data download.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pretty_midi

from deepTechno.data.preprocess import csv_to_pianoroll_dataset
from deepTechno.data.tokenizer import decode_midi, encode_midi


def _write_midi(path, notes, pedal=()):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
    for time, value in pedal:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=value, time=time))
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


def test_encode_midi_without_sustain_pedal_keeps_notes(tmp_path):
    # Techno / drum MIDI has no CC64 events; notes must not be dropped.
    path = _write_midi(tmp_path / "a.mid", [(60, 0.0, 0.5), (64, 0.5, 1.0)])
    tokens = encode_midi(path)
    assert tokens, "encode_midi returned no tokens for a pedal-free MIDI"
    decoded = decode_midi(tokens)
    assert sorted(n.pitch for n in decoded.instruments[0].notes) == [60, 64]


def test_encode_midi_keeps_notes_after_last_sustain(tmp_path):
    notes = [(60, 0.0, 0.5), (62, 1.0, 1.2), (67, 3.0, 3.5)]
    path = _write_midi(tmp_path / "b.mid", notes, pedal=[(0.9, 100), (2.0, 0)])
    decoded = decode_midi(encode_midi(path))
    assert sorted(n.pitch for n in decoded.instruments[0].notes) == [60, 62, 67]


def test_csv_windows_do_not_mix_pieces(tmp_path):
    # all_notes.csv concatenates pieces; each restarts its clock at ~0 s.
    # A global sort by start would merge piece A and piece B into one window.
    piece_a = pd.DataFrame({"pitch": [40, 41], "start": [0.0, 0.5]})
    piece_b = pd.DataFrame({"pitch": [80, 81], "start": [0.0, 0.5]})
    df = pd.concat([piece_a, piece_b], ignore_index=True)
    df["end"] = df["start"] + 0.1
    df["step"] = 0.0
    df["duration"] = 0.1
    csv = tmp_path / "notes.csv"
    df.to_csv(csv, index=False)
    rolls = csv_to_pianoroll_dataset(str(csv))
    assert rolls.shape[0] == 2
    assert rolls[0].sum() == 2 and rolls[1].sum() == 2


def test_csv_window_grid_anchored_to_window_start(tmp_path):
    # 120 BPM, 1 bar/window, 4 steps/bar -> 2 s windows, 0.5 s steps.
    # Second window starts at 2.0 s; its first note at 3.0 s must land on step 2,
    # not be re-anchored to step 0.
    df = pd.DataFrame({"pitch": [60, 60, 60], "start": [0.0, 3.0, 3.5]})
    csv = tmp_path / "notes.csv"
    df.to_csv(csv, index=False)
    rolls = csv_to_pianoroll_dataset(str(csv), n_bars=1, n_steps_per_bar=4, bpm=120.0)
    assert rolls.shape[0] == 2
    steps = np.nonzero(rolls[1].sum(axis=(0, 1, 3)))[0]
    assert steps.tolist() == [2, 3]


def test_load_model_uses_weights_only(tmp_path):
    # torch>=2.6 defaults to weights_only=True; pyproject allows torch>=2.2, so
    # load_model passes it explicitly. This pins that behaviour.
    import pytest
    torch = pytest.importorskip("torch")
    from deepTechno.generation.generate_transformer import load_model

    kw = dict(n_layers=1, num_heads=2, d_model=8, dim_feedforward=16, max_sequence=16, rpr=False)
    from deepTechno.model.transformer import MusicTransformer
    path = tmp_path / "w.pt"
    torch.save(MusicTransformer(**kw).state_dict(), path)
    assert isinstance(load_model(str(path), **kw), MusicTransformer)
    # A pickled arbitrary object (not a plain state dict) must be refused.
    bad = tmp_path / "bad.pt"
    torch.save({"x": _Evil()}, bad)
    with pytest.raises(Exception):
        load_model(str(bad), **kw)


class _Evil:
    pass
