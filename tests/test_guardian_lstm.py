"""Guardian run #2 pass 3: LSTM (TensorFlow) path. Skipped without TF."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from deepTechno.generation.generate_lstm import generate_midi_lstm  # noqa: E402
from deepTechno.model.loss import mse_with_positive_pressure  # noqa: E402


class _FakeModel:
    """Records every context window; always predicts pitch 100."""

    def __init__(self):
        self.inputs = []

    def predict(self, x, verbose=0):
        self.inputs.append(np.asarray(x))
        logits = np.full((1, 128), -1e9, dtype=np.float32)
        logits[0, 100] = 0.0
        # Keras Model.predict returns numpy arrays.
        return {"pitch": logits, "step": np.array([[0.1]], dtype=np.float32),
                "duration": np.array([[0.2]], dtype=np.float32)}


def test_generate_midi_lstm_keeps_pitch_normalised(tmp_path):
    # Training scales pitch by 1/vocab_size (create_sequences.scale_pitch); the
    # generated notes fed back as context must use the same scale.
    model = _FakeModel()
    seed = np.array([[60, 0.1, 0.2]] * 5, dtype=float)
    df = generate_midi_lstm(model, seed, str(tmp_path / "o.mid"), num_predictions=4,
                            temperature=1.0, seq_length=5)
    assert df["pitch"].tolist() == [100] * 4  # output stays in MIDI pitch units
    for ctx in model.inputs:
        assert ctx[..., 0].max() <= 1.0
    assert np.isclose(model.inputs[-1][0, -1, 0], 100 / 128)


def test_mse_with_positive_pressure_penalises_negatives():
    y = tf.constant([[1.0]])
    assert float(mse_with_positive_pressure(y, tf.constant([[1.0]]))) == 0.0
    assert float(mse_with_positive_pressure(y, tf.constant([[-1.0]]))) == pytest.approx(4.0 + 10.0)


def test_lstm_training_smoke(tmp_path):
    from deepTechno.training.train_lstm import LSTMConfig, run_lstm_training

    rng = np.random.default_rng(0)
    n = 60
    pd.DataFrame({"pitch": rng.integers(40, 80, n), "step": rng.random(n),
                  "duration": rng.random(n)}).to_csv(tmp_path / "notes.csv", index=False)
    cfg = LSTMConfig(all_notes_csv=str(tmp_path / "notes.csv"),
                     checkpoint_dir=str(tmp_path / "ckpt"), seq_length=5, batch_size=8,
                     epochs=1, lstm_units=8)
    model, hist = run_lstm_training(cfg)
    assert np.isfinite(hist.history["loss"][0])


def test_predict_next_note_rejects_non_positive_temperature():
    from deepTechno.generation.generate_lstm import predict_next_note

    with pytest.raises(ValueError):
        predict_next_note(np.zeros((5, 3)), _FakeModel(), temperature=0.0)
