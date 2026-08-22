"""PyTorch Dataset wrappers for MAESTRO."""
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from deepTechno.data.tokenizer import encode_midi
from deepTechno.model.constants import TOKEN_END, TOKEN_PAD, TORCH_LABEL_TYPE


class MaestroDataset(Dataset):
    """
    Loads MIDI files listed in a CSV and returns (input_seq, target_seq) pairs
    of event tokens for Music Transformer training.

    CSV must have a column 'midi_file' with absolute paths (regenerate from
    sourcing.find_midi_files if paths have changed).
    """

    def __init__(self, csv_path: str, max_seq: int = 2048, random_seq: bool = True):
        self.max_seq = max_seq
        self.random_seq = random_seq
        df = pd.read_csv(csv_path)
        self.data_files = df["midi_file"].tolist()

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        path = self.data_files[idx]
        try:
            raw = encode_midi(path)
        except Exception:
            raw = []

        return _process_event_tokens(raw, self.max_seq, self.random_seq)


def _process_event_tokens(raw: list, max_seq: int, random_seq: bool):
    """Trim or pad a raw token list to max_seq, return (x, y) tensors."""
    raw_len = len(raw)
    full_seq = max_seq + 1  # one extra for the shifted target

    if raw_len == 0:
        x = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
        y = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
        return x, y

    if raw_len < full_seq:
        tokens = raw + [TOKEN_END] + [TOKEN_PAD] * (full_seq - raw_len - 1)
    elif random_seq:
        start = random.randint(0, raw_len - full_seq)
        tokens = raw[start: start + full_seq]
    else:
        tokens = raw[:full_seq]

    x = torch.tensor(tokens[:-1], dtype=TORCH_LABEL_TYPE)
    y = torch.tensor(tokens[1:],  dtype=TORCH_LABEL_TYPE)
    return x, y


def compute_epiano_accuracy(y_hat, y_gt):
    """Token-level accuracy ignoring PAD positions."""
    softmax = torch.softmax(y_hat, dim=-1)
    predicted = torch.argmax(softmax, dim=-1)
    mask = y_gt != TOKEN_PAD
    correct = (predicted[mask] == y_gt[mask]).sum().float()
    return correct / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
