"""LSTM generation: temperature-sampled note prediction."""
import numpy as np
import tensorflow as tf

from deepTechno.data.preprocess import notes_to_midi


def predict_next_note(notes: np.ndarray, model: tf.keras.Model,
                      temperature: float = 1.0):
    """Return (pitch, step, duration) for the next note given a context window."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs, verbose=0)
    pitch_logits = predictions["pitch"] / temperature
    # Reduce to true scalars: int()/float() of a shape-(1,) tensor raises
    # TypeError on NumPy 2 ("only 0-dimensional arrays can be converted").
    pitch = int(tf.reshape(tf.random.categorical(pitch_logits, num_samples=1), [-1])[0])
    step     = max(0.0, float(np.reshape(predictions["step"],     -1)[0]))
    duration = max(0.0, float(np.reshape(predictions["duration"], -1)[0]))
    return pitch, step, duration


def generate_midi_lstm(model: tf.keras.Model, seed_notes: np.ndarray,
                       out_file: str, instrument_name: str = "Acoustic Grand Piano",
                       num_predictions: int = 120, temperature: float = 2.0,
                       seq_length: int = 25, vocab_size: int = 128):
    """
    Generate a MIDI file from a trained LSTM model.

    Args:
        model:            trained Keras model with pitch/step/duration outputs
        seed_notes:       numpy array shape (N, 3) — raw (pitch, step, duration) rows
        out_file:         output .mid path
        instrument_name:  pretty_midi instrument name
        num_predictions:  number of notes to generate
        temperature:      sampling temperature (higher = more random)
        seq_length:       model input window length
        vocab_size:       MIDI pitch range for normalisation
    """
    import pandas as pd

    key_order = ["pitch", "step", "duration"]
    input_notes = seed_notes[:seq_length] / np.array([vocab_size, 1, 1])

    generated = []
    prev_start = 0.0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        generated.append((pitch, step, duration, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        # Feed the context back on the training scale (pitch / vocab_size, see
        # sourcing.create_sequences); the raw pitch goes only into the output.
        next_input = (pitch / vocab_size, step, duration)
        input_notes = np.append(input_notes, np.expand_dims(next_input, 0), axis=0)
        prev_start = start

    generated_df = pd.DataFrame(generated, columns=[*key_order, "start", "end"])
    notes_to_midi(generated_df, out_file=out_file, instrument_name=instrument_name)
    print(f"Generated {num_predictions} notes → {out_file}")
    return generated_df
