"""MAESTRO dataset download, file discovery, and train/val/test splitting."""
import csv
import os
import zipfile
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # TF is imported lazily inside the functions that need it
    import tensorflow as tf

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"


def download_maestro(data_dir: str) -> str:
    """Download and extract MAESTRO v2.0.0 MIDI zip if not already present. Returns data_dir."""
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"MAESTRO already present at {data_dir}")
        return data_dir

    import tensorflow as tf  # lazy: the CSV/zip helpers work without TF

    zip_path = tf.keras.utils.get_file(
        "maestro-v2.0.0-midi.zip",
        origin=MAESTRO_URL,
        extract=False,
        cache_dir=os.path.dirname(data_dir),
        cache_subdir=os.path.basename(data_dir),
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        safe_extract(zf, data_dir)
    return data_dir


def safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    """Extract ``zf`` into ``dest``, refusing members that would land outside it.

    Checks every member before writing anything (zip-slip defence in depth on
    top of ``extractall``'s own sanitising).
    """
    root = os.path.realpath(dest)
    for name in zf.namelist():
        if os.path.isabs(name) or name.startswith(("/", "\\")):
            raise ValueError(f"Refusing absolute path in archive: {name!r}")
        target = os.path.realpath(os.path.join(root, name))
        if os.path.commonpath([root, target]) != root:
            raise ValueError(f"Refusing path outside {dest!r}: {name!r}")
    zf.extractall(root)


def find_midi_files(dataset_folder: str, csv_filename: str):
    """Walk dataset_folder for MIDI files, write paths to csv_filename. Returns (list, csv_path)."""
    midi_files = []
    for root, _, files in os.walk(dataset_folder):
        for f in files:
            if f.endswith(".midi") or f.endswith(".mid"):
                midi_files.append(os.path.join(root, f))

    csv_path = os.path.join(dataset_folder, csv_filename)
    if midi_files:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["midi_file"])
            for p in midi_files:
                writer.writerow([p])
    else:
        print("No MIDI files found.")
    return midi_files, csv_path


def split_csv(csv_file: str, train_csv: str, val_csv: str, test_csv: str,
              train_p=0.6, val_p=0.2, test_p=0.2, data_dir="."):
    """Split a CSV of file paths into train/val/test CSVs."""
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_p)
    n_val   = int(n * val_p)

    df.iloc[:n_train].to_csv(os.path.join(data_dir, train_csv), index=False)
    df.iloc[n_train:n_train + n_val].to_csv(os.path.join(data_dir, val_csv), index=False)
    df.iloc[n_train + n_val:].to_csv(os.path.join(data_dir, test_csv), index=False)
    print(f"Split: {n_train} train / {n_val} val / {n - n_train - n_val} test")


def create_sequences(dataset: "tf.data.Dataset", seq_length: int,
                     vocab_size: int = 128, key_order=None) -> "tf.data.Dataset":
    """Sliding-window sequences + label split for LSTM training."""
    if key_order is None:
        key_order = ["pitch", "step", "duration"]

    import tensorflow as tf

    windows = dataset.window(seq_length + 1, shift=1, stride=1, drop_remainder=True)

    def flatten(x):
        return x.batch(seq_length + 1, drop_remainder=True)

    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        return x / [vocab_size, 1.0, 1.0]

    def split_labels(seq):
        inputs = seq[:-1]
        labels_dense = seq[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
