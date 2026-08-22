"""MIDI ↔ note DataFrame conversion (used by the LSTM track)."""
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
