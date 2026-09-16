"""MIDI ↔ note DataFrame conversion (used by the LSTM track).

Also hosts the ``all_notes.csv`` -> multi-track pianoroll bridge that feeds the
MuseGAN subsystem (:mod:`deepTechno.musegan`). See :func:`notes_to_pianoroll`.
"""
from __future__ import annotations

import collections
import numpy as np
import pandas as pd
import pretty_midi


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """Parse a MIDI file into a DataFrame with columns pitch, start, end, step, duration."""
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        notes["pitch"].append(note.pitch)
        notes["start"].append(note.start)
        notes["end"].append(note.end)
        notes["step"].append(note.start - prev_start)
        notes["duration"].append(note.end - note.start)
        prev_start = note.start

    return pd.DataFrame({k: np.array(v) for k, v in notes.items()})


def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str,
                  velocity: int = 100) -> pretty_midi.PrettyMIDI:
    """Convert a note DataFrame back to a MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note["step"])
        end = float(start + note["duration"])
        instrument.notes.append(pretty_midi.Note(
            velocity=velocity, pitch=int(note["pitch"]), start=start, end=end))
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def get_note_names(pitch_array: np.ndarray) -> np.ndarray:
    return np.vectorize(pretty_midi.note_number_to_name)(pitch_array)


# --------------------------------------------------------------------------- #
# all_notes.csv -> multi-track pianoroll (MuseGAN bridge)
# --------------------------------------------------------------------------- #
# The MuseGAN generator/critic operate on a binary pianoroll tensor with shape
# (n_tracks, n_bars, n_steps_per_bar, n_pitches). all_notes.csv is a flat note
# table (columns: pitch, start, end, step, duration; start/end in seconds), so
# we quantise onto a techno-appropriate straight 4/4 grid and split notes into
# tracks by pitch register (the CSV has no per-track channel).

# Default geometry, matching MuseGenerator/MuseCritic defaults.
DEFAULT_N_TRACKS = 4
DEFAULT_N_BARS = 2
DEFAULT_N_STEPS_PER_BAR = 16  # 16 steps/bar = 16th-note grid in 4/4 (techno)
DEFAULT_N_PITCHES = 84
DEFAULT_PITCH_LOW = 24  # MIDI note range covered by the roll: [24, 24+84)


def _assign_track(pitch: int, n_tracks: int, pitch_low: int, n_pitches: int) -> int:
    """Assign a note to a track by pitch register (low register -> track 0)."""
    span = max(n_pitches // n_tracks, 1)
    rel = int(pitch) - pitch_low
    return min(max(rel // span, 0), n_tracks - 1)


def notes_to_pianoroll(
    notes: pd.DataFrame,
    n_tracks: int = DEFAULT_N_TRACKS,
    n_bars: int = DEFAULT_N_BARS,
    n_steps_per_bar: int = DEFAULT_N_STEPS_PER_BAR,
    n_pitches: int = DEFAULT_N_PITCHES,
    pitch_low: int = DEFAULT_PITCH_LOW,
    bpm: float = 128.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Quantise a note DataFrame into a binary multi-track pianoroll tensor.

    Parameters
    ----------
    notes:
        DataFrame with at least ``pitch`` and ``start`` columns (seconds), as in
        ``all_notes.csv`` / :func:`midi_to_notes`.
    n_tracks, n_bars, n_steps_per_bar, n_pitches:
        Output geometry. Must match the MuseGAN generator/critic.
    pitch_low:
        Lowest MIDI pitch mapped to index 0; pitches outside
        ``[pitch_low, pitch_low + n_pitches)`` are dropped.
    bpm:
        Assumed tempo for seconds -> step quantisation (techno default 128 BPM,
        straight 4/4). One bar = ``4 * 60 / bpm`` seconds.
    dtype:
        Output dtype (float32 for direct feeding to torch).

    Returns
    -------
    np.ndarray
        Binary tensor of shape ``(n_tracks, n_bars, n_steps_per_bar, n_pitches)``.
    """
    roll = np.zeros((n_tracks, n_bars, n_steps_per_bar, n_pitches), dtype=dtype)
    if len(notes) == 0:
        return roll

    sec_per_bar = 4.0 * 60.0 / bpm
    sec_per_step = sec_per_bar / n_steps_per_bar
    total_steps = n_bars * n_steps_per_bar

    starts = notes["start"].to_numpy(dtype=float)
    pitches = notes["pitch"].to_numpy()
    origin = float(starts.min())

    for start, pitch in zip(starts, pitches):
        step_idx = int(round((start - origin) / sec_per_step))
        if step_idx < 0 or step_idx >= total_steps:
            continue
        pitch_idx = int(pitch) - pitch_low
        if pitch_idx < 0 or pitch_idx >= n_pitches:
            continue
        bar = step_idx // n_steps_per_bar
        step = step_idx % n_steps_per_bar
        track = _assign_track(int(pitch), n_tracks, pitch_low, n_pitches)
        roll[track, bar, step, pitch_idx] = 1.0

    return roll


def csv_to_pianoroll_dataset(
    csv_path: str,
    max_windows: int | None = None,
    n_tracks: int = DEFAULT_N_TRACKS,
    n_bars: int = DEFAULT_N_BARS,
    n_steps_per_bar: int = DEFAULT_N_STEPS_PER_BAR,
    n_pitches: int = DEFAULT_N_PITCHES,
    pitch_low: int = DEFAULT_PITCH_LOW,
    bpm: float = 128.0,
) -> np.ndarray:
    """Slice ``all_notes.csv`` into a batch of pianoroll windows.

    Notes are read in ``start`` order and chunked into consecutive
    ``n_bars``-long windows, each quantised by :func:`notes_to_pianoroll`.

    Returns
    -------
    np.ndarray
        Shape ``(n_windows, n_tracks, n_bars, n_steps_per_bar, n_pitches)``.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("start").reset_index(drop=True)
    sec_per_window = n_bars * 4.0 * 60.0 / bpm
    origin = float(df["start"].min())
    df = df.assign(_window=((df["start"] - origin) // sec_per_window).astype(int))

    windows = []
    for _, group in df.groupby("_window"):
        windows.append(
            notes_to_pianoroll(
                group, n_tracks=n_tracks, n_bars=n_bars,
                n_steps_per_bar=n_steps_per_bar, n_pitches=n_pitches,
                pitch_low=pitch_low, bpm=bpm,
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break

    if not windows:
        return np.zeros((0, n_tracks, n_bars, n_steps_per_bar, n_pitches), dtype=np.float32)
    return np.stack(windows, axis=0)
