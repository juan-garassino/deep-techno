"""
Text encoder for composer/style-conditioned MIDI generation.

Encodes a text string (e.g. "Chopin" or "Chopin – Nocturne Op. 9") into a
sequence of prefix tokens that are prepended to the MIDI token sequence before
feeding into the Music Transformer decoder.

Vocabulary is built from the MAESTRO canonical_composer + canonical_title fields
(~200 unique composers, ~1,200 unique titles). No external LLM required.
"""
import json
import os
import re
from typing import List, Optional

import torch
import torch.nn as nn

# Special tokens
PAD_TOKEN  = "<pad>"
UNK_TOKEN  = "<unk>"
BOS_TOKEN  = "<bos>"  # start of text prefix
EOS_TOKEN  = "<eos>"  # end of text prefix


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower().strip()
    tokens = re.findall(r"[a-z0-9]+|[^\w\s]", text)
    return tokens


class TextVocab:
    """Char/word vocabulary built from MAESTRO metadata strings."""

    SPECIALS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self):
        self.token2idx = {t: i for i, t in enumerate(self.SPECIALS)}
        self.idx2token = list(self.SPECIALS)

    def build_from_strings(self, strings: List[str]):
        for s in strings:
            for tok in _tokenize(s):
                if tok not in self.token2idx:
                    self.token2idx[tok] = len(self.idx2token)
                    self.idx2token.append(tok)

    def encode(self, text: str, max_len: int = 32) -> List[int]:
        tokens = [BOS_TOKEN] + _tokenize(text)[:max_len - 2] + [EOS_TOKEN]
        return [self.token2idx.get(t, self.token2idx[UNK_TOKEN]) for t in tokens]

    def __len__(self):
        return len(self.idx2token)

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump({"token2idx": self.token2idx, "idx2token": self.idx2token}, fh)

    @classmethod
    def load(cls, path: str) -> "TextVocab":
        vocab = cls()
        with open(path) as fh:
            data = json.load(fh)
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = data["idx2token"]
        return vocab


class TextEncoder(nn.Module):
    """
    Embeds a text token sequence into d_model-dimensional prefix vectors.

    Usage in training:
        text_ids  = vocab.encode("Chopin")                   # list[int]
        text_ids  = torch.tensor(text_ids).unsqueeze(0)      # (1, T_text)
        prefix    = encoder(text_ids)                        # (1, T_text, d_model)
        # Concatenate prefix with MIDI embeddings before the transformer decoder.

    Usage in generation:
        Pass prefix as the first tokens of the sequence (causal mask handles the rest).
    """

    def __init__(self, vocab_size: int, d_model: int = 512,
                 max_text_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_text_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_text_len = max_text_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) integer tensor
        Returns:
            (B, T, d_model) prefix embeddings
        """
        T = token_ids.size(1)
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.embedding(token_ids) + self.pos_embedding(positions)
        return self.dropout(x)


def build_vocab_from_maestro(maestro_csv_path: str,
                              save_path: Optional[str] = None) -> TextVocab:
    """
    Build a TextVocab from the MAESTRO metadata CSV.

    Args:
        maestro_csv_path: path to maestro-v2.0.0.csv
        save_path:        if given, save the vocab to this JSON path

    Returns:
        TextVocab instance
    """
    import pandas as pd
    df = pd.read_csv(maestro_csv_path)
    strings = []
    if "canonical_composer" in df.columns:
        strings += df["canonical_composer"].dropna().tolist()
    if "canonical_title" in df.columns:
        strings += df["canonical_title"].dropna().tolist()

    vocab = TextVocab()
    vocab.build_from_strings(strings)
    print(f"TextVocab built: {len(vocab)} tokens from {len(strings)} strings")

    if save_path:
        vocab.save(save_path)
        print(f"Vocab saved to {save_path}")

    return vocab
