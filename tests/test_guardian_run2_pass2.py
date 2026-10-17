"""Guardian run #2 pass 2: sourcing (zip-slip, CSV split), text vocab,
loss edge case and an end-to-end generation smoke test. No TF, no network."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pretty_midi
import pytest


# --------------------------------------------------------------------------- #
# sourcing.py (must import without TensorFlow)
# --------------------------------------------------------------------------- #

def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf


def test_safe_extract_rejects_zip_slip(tmp_path):
    from deepTechno.data.sourcing import safe_extract

    dest = tmp_path / "data"
    dest.mkdir()
    for evil in ("../escape.txt", "a/../../escape.txt", "/abs/escape.txt"):
        with zipfile.ZipFile(_zip_bytes({"ok.mid": b"x", evil: b"pwn"})) as zf:
            with pytest.raises(ValueError):
                safe_extract(zf, str(dest))
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_extracts_normal_members(tmp_path):
    from deepTechno.data.sourcing import safe_extract

    with zipfile.ZipFile(_zip_bytes({"maestro/2004/a.midi": b"x"})) as zf:
        safe_extract(zf, str(tmp_path))
    assert (tmp_path / "maestro/2004/a.midi").read_bytes() == b"x"


def test_find_midi_files_and_split_csv(tmp_path):
    from deepTechno.data.sourcing import find_midi_files, split_csv

    for i in range(10):
        (tmp_path / f"{i}.mid").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("not midi")
    files, csv_path = find_midi_files(str(tmp_path), "all.csv")
    assert len(files) == 10
    split_csv("all.csv", "tr.csv", "va.csv", "te.csv", data_dir=str(tmp_path))
    parts = [pd.read_csv(tmp_path / n)["midi_file"].tolist() for n in ("tr.csv", "va.csv", "te.csv")]
    assert [len(p) for p in parts] == [6, 2, 2]
    assert sorted(sum(parts, [])) == sorted(files)


# --------------------------------------------------------------------------- #
# text encoder vocab
# --------------------------------------------------------------------------- #

def test_tokenize_keeps_accented_words():
    pytest.importorskip("torch")
    from deepTechno.encoders.text_encoder import _tokenize

    # MAESTRO composers include e.g. Frédéric Chopin, Antonín Dvořák.
    assert _tokenize("Frédéric Chopin") == ["frédéric", "chopin"]
    assert _tokenize("Antonín Dvořák, Op. 9") == ["antonín", "dvořák", ",", "op", ".", "9"]


def test_build_vocab_from_maestro_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from deepTechno.encoders.text_encoder import (
        TextEncoder, TextVocab, UNK_TOKEN, build_vocab_from_maestro)

    csv = tmp_path / "maestro.csv"
    pd.DataFrame({"canonical_composer": ["Frédéric Chopin", "Franz Liszt", None],
                  "canonical_title": ["Nocturne Op. 9", "Étude", "Waltz"]}).to_csv(csv, index=False)
    vocab = build_vocab_from_maestro(str(csv), save_path=str(tmp_path / "v.json"))
    ids = vocab.encode("Chopin Nocturne")
    assert vocab.token2idx[UNK_TOKEN] not in ids
    assert vocab.encode("Bach")[1] == vocab.token2idx[UNK_TOKEN]
    loaded = TextVocab.load(str(tmp_path / "v.json"))
    assert loaded.encode("Chopin Nocturne") == ids and len(loaded) == len(vocab)

    enc = TextEncoder(len(vocab), d_model=8, max_text_len=32)
    out = enc(torch.tensor([ids]))
    assert out.shape == (1, len(ids), 8)


# --------------------------------------------------------------------------- #
# loss + generation
# --------------------------------------------------------------------------- #

def test_smooth_ce_all_targets_ignored_is_finite():
    torch = pytest.importorskip("torch")
    from deepTechno.model.loss import SmoothCrossEntropyLoss

    loss = SmoothCrossEntropyLoss(0.1, vocab_size=5, ignore_index=4)
    val = loss(torch.randn(3, 5), torch.tensor([4, 4, 4]))
    assert torch.isfinite(val) and val.item() == 0.0


def test_generate_from_dataset_short_primer_writes_midi(tmp_path):
    # Short file -> primer padded with END/PAD -> generate -> decode -> write.
    pytest.importorskip("torch")
    from deepTechno.generation.generate_transformer import generate_from_dataset
    from deepTechno.model.transformer import MusicTransformer

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    mid = tmp_path / "a.mid"
    pm.write(str(mid))
    csv = tmp_path / "ds.csv"
    pd.DataFrame({"midi_file": [str(mid)]}).to_csv(csv, index=False)

    model = MusicTransformer(n_layers=1, num_heads=2, d_model=8, dim_feedforward=16,
                             max_sequence=32, rpr=False).eval()
    tokens = generate_from_dataset(model, str(csv), idx=0, num_primer=8,
                                   target_seq_length=16, out_dir=str(tmp_path / "gen"))
    assert 8 <= len(tokens) <= 16
    written = pretty_midi.PrettyMIDI(str(tmp_path / "gen" / "rand.mid"))
    assert 60 in [n.pitch for n in written.instruments[0].notes]


@pytest.mark.parametrize("rpr", [False, True])
def test_music_transformer_forward_on_current_torch(rpr):
    # nn.Transformer passes tgt_is_causal / is_causal kwargs since torch 2.1;
    # the custom decoder/encoder must accept them.
    torch = pytest.importorskip("torch")
    from deepTechno.model.constants import VOCAB_SIZE
    from deepTechno.model.transformer import MusicTransformer

    model = MusicTransformer(n_layers=1, num_heads=2, d_model=8, dim_feedforward=16,
                             max_sequence=16, rpr=rpr)
    y = model(torch.randint(0, 300, (2, 10)))
    assert y.shape == (2, 10, VOCAB_SIZE)


def test_transformer_training_smoke_and_resume_without_epoch(tmp_path):
    # Tiny end-to-end run (1 epoch, CPU) that writes checkpoints, then resumes
    # from one with continue_weights but no continue_epoch.
    pytest.importorskip("torch")
    from deepTechno.training.train_transformer import TransformerConfig, run_transformer_training

    paths = []
    for i in range(2):
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for k in range(6):
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=60 + k + i,
                                               start=0.25 * k, end=0.25 * k + 0.2))
        pm.instruments.append(inst)
        p = tmp_path / f"{i}.mid"
        pm.write(str(p))
        paths.append(str(p))
    csv = tmp_path / "ds.csv"
    pd.DataFrame({"midi_file": paths}).to_csv(csv, index=False)

    cfg = dict(train_csv=str(csv), val_csv=str(csv), output_dir=str(tmp_path / "out"),
               n_layers=1, num_heads=2, d_model=8, dim_feedforward=16, max_sequence=32,
               rpr=False, batch_size=2, epochs=1, n_workers=0, weight_modulus=1,
               no_tensorboard=True, force_cpu=True)
    run_transformer_training(TransformerConfig(**cfg))
    ckpt = tmp_path / "out" / "weights" / "epoch_0001.pt"
    assert ckpt.exists()

    resumed = run_transformer_training(TransformerConfig(**{**cfg, "continue_weights": str(ckpt)}))
    assert resumed is not None
