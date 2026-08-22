from deepTechno.preprocess.midi_encoder_decoder import encode_midi
from deepTechno.manager.manager import Manager
import os
import pretty_midi
import collections
import pandas as pd
import numpy as np

def print_note_info(instrument):
    """
    Print information about notes in the instrument.

    Args:
    - instrument (pretty_midi.Instrument): Instrument object from PrettyMIDI.
    """
    for i, note in enumerate(instrument.notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        print(f'{i}: pitch={note.pitch}, note_name={note_name}, duration={duration:.4f}')

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """
    Convert MIDI file to DataFrame containing note information.

    Args:
    - midi_file (str): Path to the MIDI file.

    Returns:
    - pd.DataFrame: DataFrame containing note information (pitch, start time, end time, step, duration).
    """
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def get_note_names(pitch_array):
    """
    Convert MIDI pitch numbers to note names.

    Args:
    - pitch_array (np.array): Array of MIDI pitch numbers.

    Returns:
    - np.array: Array of corresponding note names.
    """
    get_note_name = np.vectorize(pretty_midi.note_number_to_name)
    return get_note_name(pitch_array)

def find_unique_notes(dataset_folder):
    unique_notes = set()

    # Iterate over all MIDI files in the dataset folder
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                midi_path = os.path.join(root, file)
                pm = pretty_midi.PrettyMIDI(midi_path)

                # Extract notes from each MIDI file
                for instrument in pm.instruments:
                    for note in instrument.notes:
                        # Add each note to the set of unique notes
                        unique_notes.add(note.pitch)

    return unique_notes

if __name__ == "__main__":
    # Example usage:
    midi = './dataset/maestro-v2.0.0/2013/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--2.midi'
    # Print note information for the first 10 notes in an instrument
    pm = pretty_midi.PrettyMIDI(midi)
    
    instrument = pm.instruments[0]
    
    print_note_info(instrument)

    # Convert MIDI file to DataFrame containing note information
    notes_df = midi_to_notes(midi)
    
    print(notes_df.head())

    # Convert MIDI pitch numbers to note names
    pitch_array = np.array([60, 62, 64, 65, 67, 69, 71, 72])
    
    note_names = get_note_names(pitch_array)
    
    print(note_names)

    # temps = find_unique_notes('./dataset/maestro-v2.0.0')
    
    # print(temps)