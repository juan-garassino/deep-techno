"""LSTM training (TensorFlow/Keras) on MAESTRO note sequences."""
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from deepTechno.data.sourcing import create_sequences
from deepTechno.model.loss import mse_with_positive_pressure


@dataclass
class LSTMConfig:
    all_notes_csv: str = "all_notes.csv"
    checkpoint_dir: str = "./results/training_checkpoints"
    seq_length: int = 25
    vocab_size: int = 128
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.005
    lstm_units: int = 128
    key_order: List[str] = field(default_factory=lambda: ["pitch", "step", "duration"])


def build_lstm_model(config: LSTMConfig) -> tf.keras.Model:
    inputs = tf.keras.Input((config.seq_length, 3))
    x = tf.keras.layers.LSTM(config.lstm_units)(inputs)
    outputs = {
        "pitch":    tf.keras.layers.Dense(config.vocab_size, name="pitch")(x),
        "step":     tf.keras.layers.Dense(1, name="step")(x),
        "duration": tf.keras.layers.Dense(1, name="duration")(x),
    }
    return tf.keras.Model(inputs, outputs)


def run_lstm_training(config: LSTMConfig):
    # ── Load notes ────────────────────────────────────────────────
    all_notes = pd.read_csv(config.all_notes_csv)
    n_notes = len(all_notes)
    print(f"Notes loaded: {n_notes:,}")

    train_notes = np.stack([all_notes[k] for k in config.key_order], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    seq_ds = create_sequences(notes_ds, config.seq_length, config.vocab_size, config.key_order)

    buffer_size = n_notes - config.seq_length
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(config.batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.AUTOTUNE))

    # ── Build model ───────────────────────────────────────────────
    model = build_lstm_model(config)
    loss = {
        "pitch":    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "step":     mse_with_positive_pressure,
        "duration": mse_with_positive_pressure,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        loss=loss,
        loss_weights={"pitch": 0.05, "step": 1.0, "duration": 1.0},
        optimizer=optimizer,
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_dir, "ckpt_{epoch}.weights.h5"),
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
    ]

    history = model.fit(train_ds, epochs=config.epochs, callbacks=callbacks)

    # Save final checkpoint via tf.train.CheckpointManager
    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, config.checkpoint_dir, max_to_keep=5)
    manager.save()

    return model, history
