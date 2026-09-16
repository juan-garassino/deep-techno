# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**AUD-deep-techno** is a multi-modal MIDI music generation repo. It has a shared decoder-only Music Transformer (PyTorch) that can be conditioned on different input modalities, plus a parallel LSTM baseline (TensorFlow/Keras).

| Modality | Status | Notebook |
|---|---|---|
| Unconditional MIDI continuation | Working | `03_train_transformer.ipynb` |
| Text → MIDI (composer/style prefix) | Working | `04_train_text_conditioned.ipynb` |
| LSTM (pitch/step/duration heads) | Working | `02_train_lstm.ipynb` |
| Vision → MIDI (sheet music OMR) | Scaffolded | `05_vision_to_midi.ipynb` (Phase 2) |
| MuseGAN (multi-track WGAN-GP pianoroll) | Scaffolded — **NEEDS-GPU-VALIDATION** | — (see `src/deepTechno/musegan/`) |

## Package structure

```
src/deepTechno/
  data/
    preprocess.py          # midi_to_notes(), notes_to_midi();
                           #   notes_to_pianoroll(), csv_to_pianoroll_dataset() (MuseGAN bridge)
    dataset.py             # MaestroDataset (PyTorch), compute_epiano_accuracy()
    sourcing.py            # download_maestro(), find_midi_files(), split_csv(), create_sequences()
    tokenizer.py           # encode_midi(), decode_midi() — event-based MIDI vocab
  encoders/
    text_encoder.py        # TextVocab, TextEncoder, build_vocab_from_maestro()
    vision_encoder.py      # VisionEncoder (ViT-style patch encoder — Phase 2)
  model/
    transformer.py         # MusicTransformer, DummyDecoder
    rpr.py                 # Relative Position Representation attention (Shaw et al. 2018)
    positional_encoding.py # Sinusoidal PE
    loss.py                # SmoothCrossEntropyLoss, mse_with_positive_pressure
    constants.py           # TOKEN_END, TOKEN_PAD, VOCAB_SIZE, RANGE_*, Adam params
  training/
    train_lstm.py          # LSTMConfig, build_lstm_model(), run_lstm_training()
    train_transformer.py   # TransformerConfig, train_epoch(), eval_model(), run_transformer_training()
  generation/
    generate_lstm.py       # predict_next_note(), generate_midi_lstm()
    generate_transformer.py # load_model(), generate_from_primer(), generate_from_file(), generate_from_dataset()
  musegan/                 # multi-track WGAN-GP — NEEDS-GPU-VALIDATION (scaffold only)
    generator.py           # MuseGenerator (4-way latent), TemporalNetwork, BarGenerator, Reshape
    critic.py              # MuseCritic (3D-conv WGAN critic)
    binary_neuron.py       # BinaryNeuron STE output layer (salu133445 concept, PyTorch autograd.Function)
    training_loop.py       # MuseGanTrainer (WGAN-GP), WassersteinLoss, GradientPenalty
  utils/
    device.py              # get_device(), use_cuda()
    lr_scheduling.py       # LrStepTracker, get_lr()
_legacy/                   # original deeptechno/, deepTechno1/, pyproject.toml, poetry.lock
notebooks/                 # Colab-ready .ipynb files (00–04)
dataset/                   # MAESTRO CSVs (paths are Colab-relative after running 00_setup)
results/                   # checkpoints, generated MIDI
```

## Commands (local dev)

```bash
# Install (setuptools, no poetry)
pip install -e .
pip install -e ".[vision]"   # include timm for Phase 2 vision encoder

# Smoke test imports
python -c "from deepTechno.model.transformer import MusicTransformer; print('ok')"
python -c "from deepTechno.encoders.text_encoder import TextEncoder; print('ok')"
python -c "from deepTechno.musegan import MuseGenerator, MuseCritic, MuseGanTrainer; print('ok')"

# MuseGAN (needs torch: pip install -e ".[musegan]"). Shape tests only — no training here:
pytest tests/test_musegan.py -q

# Run LSTM training locally
python -c "
from deepTechno.training.train_lstm import LSTMConfig, run_lstm_training
run_lstm_training(LSTMConfig(all_notes_csv='all_notes.csv', epochs=2))
"

# Run Transformer training locally
python -c "
from deepTechno.training.train_transformer import TransformerConfig, run_transformer_training
run_transformer_training(TransformerConfig(train_csv='dataset/train.csv', val_csv='dataset/val.csv', epochs=1, batch_size=1, no_tensorboard=True))
"
```

## Colab data strategy

Three options for getting `all_notes.csv` (953MB) into Colab — see `notebooks/00_setup_and_data.ipynb`:

| Option | How | Persistence |
|---|---|---|
| **A** — Upload from local machine | `google.colab.files.upload()` | Session only |
| **B** — Google Drive (recommended) | Upload once to `MyDrive/deep-techno-data/`, read in-place | Permanent |
| **C** — Re-parse from MAESTRO | Runs `midi_to_notes()` over all MIDIs (~15 min) | Ephemeral |

**MAESTRO MIDI** (57MB zip) is always re-downloaded from `storage.googleapis.com/magentadata` (~6 sec on Colab, Google infra). Split CSVs are regenerated each session via `sourcing.find_midi_files()` + `split_csv()` — the old ones in `dataset/` have hardcoded local paths and are unusable in Colab.

**Checkpoints** are written directly to `MyDrive/deep-techno-data/checkpoints/` (no copy needed).

## Key constants

The MIDI event vocabulary (in `model/constants.py`):
- `VOCAB_SIZE = 388` (128 NOTE_ON + 128 NOTE_OFF + 32 VEL + 100 TIME_SHIFT + 1 END + 1 PAD)
- `TOKEN_END = 388`, `TOKEN_PAD = 389`

## Architecture notes

- **MusicTransformer**: decoder-only via masked encoder stack + `DummyDecoder`. Set `rpr=True` for Relative Position Representations.
- **LSTM**: TF/Keras, multi-output (pitch softmax + step/duration regression with positive-pressure MSE loss).
- **Text conditioning**: `TextEncoder` embeds composer/title BPE tokens into `d_model`-dimensional prefix. Prepend to MIDI token embeddings before the transformer. Vocab built from MAESTRO metadata (~200 composers). No external LLM.
- **Vision conditioning** (Phase 2): `VisionEncoder` patch-projects sheet music images → encoder memory → cross-attention in decoder. Target dataset: PrIMuS (87k PNG + MIDI pairs).
- **MuseGAN** (`musegan/`, PyTorch): ported from akanametov/musegan (generator/critic + WGAN-GP) with the binary-neuron STE *concept* from salu133445/musegan reimplemented via `torch.autograd.Function`. The generator takes a **4-way latent** — `forward(chords, style, melody, groove)` — where chords/style are shared across tracks (chords time-varying via `TemporalNetwork`, style static) and melody/groove are per-track (melody time-varying, groove static). Output pianoroll shape `(batch, n_tracks, n_bars, n_steps_per_bar, n_pitches)`; techno default `4 × 2 × 16 × 84` (16-step 4/4 grid). `MuseGanTrainer` runs `repeat` critic steps per generator step with a gradient-penalty of weight 10.
  - **Sizing gotcha**: `BarGenerator` has a fixed tconv chain requiring `hid_features // hid_channels == 2`. Via `MuseGenerator` this means passing `hid_features == hid_channels`; a `ValueError` is raised otherwise.

## all_notes.csv → MuseGAN bridge

The MuseGAN subsystem consumes multi-track pianoroll tensors, but `all_notes.csv` (14.2M rows, columns `pitch,start,end,step,duration`, seconds) is a flat note table. `data/preprocess.py` bridges the two:

- `notes_to_pianoroll(notes, ...)` — quantise one window of notes onto a straight 4/4 grid (default 128 BPM), returning a binary `(n_tracks, n_bars, n_steps_per_bar, n_pitches)` tensor. Since the CSV has no per-track channel, notes are **split into tracks by pitch register**.
- `csv_to_pianoroll_dataset(csv_path, max_windows=...)` — slice the CSV into consecutive `n_bars`-long windows, returning `(n_windows, n_tracks, n_bars, n_steps_per_bar, n_pitches)`.

## NEEDS-GPU-VALIDATION

The MuseGAN subsystem is a **scaffold**: fully wired (generator, critic, WGAN-GP loop, STE) and **shape/import-tested on CPU** (`tests/test_musegan.py`), but **no real training has run**. Before use, validate on a GPU (RunPod, `garassino-ml`): (1) train `MuseGanTrainer` on `csv_to_pianoroll_dataset(all_notes.csv)` for a few epochs; (2) confirm losses are finite and the critic separates real/fake; (3) tune the pitch-register → track assignment and BPM to the actual dataset; (4) add the salu133445 music-metrics eval battery (empty-bar ratio, used-pitch-classes, qualified-note rate) as a follow-up.

## Sibling audio repos

| Repo | Role |
|---|---|
| `AUD-partiture-to-midi` | Phase 2 entry point for sheet music OMR; actual model code lives here in `encoders/vision_encoder.py` |
| `AUD-audio-transformer` | Future: audio → MIDI transcription (placeholder) |
| `AUD-stereo-boost` | Separate product (music personalization); not merged here |
