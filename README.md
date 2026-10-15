# deep-techno

Multi-modal MIDI generation in one repo: an LSTM baseline, a Music Transformer with optional text conditioning, and a scaffolded multi-track MuseGAN.

## What and why

The goal is to generate symbolic music (MIDI) and to steer it with other inputs. A shared decoder-only Music Transformer (PyTorch, with relative-position attention) is trained on the MAESTRO piano dataset and can be given a text prefix such as a composer name. A Keras LSTM serves as a baseline, and a MuseGAN-style WGAN-GP targets multi-track techno pianorolls. Training is meant to run in Google Colab through the numbered notebooks, with the package in `src/` doing the work.

## Highlights

- **Music Transformer** with an event-based MIDI vocabulary (note on/off, velocity, time shift, end, pad) and Shaw et al. relative position attention (`src/deepTechno/model/`).
- **Text-conditioned generation**: a `TextEncoder` and a vocabulary built from MAESTRO metadata, demonstrated in `notebooks/04_train_text_conditioned.ipynb`.
- **LSTM baseline** in TensorFlow/Keras with pitch, step and duration heads (`src/deepTechno/training/train_lstm.py`).
- **MuseGAN scaffold**: 4-way latent generator, 3D-conv critic, binary-neuron straight-through output and a WGAN-GP trainer, covered by shape tests (`src/deepTechno/musegan/`).
- **CSV to pianoroll bridge** that quantises a flat note table onto a 4/4 grid for the GAN (`src/deepTechno/data/preprocess.py`).

## Quickstart

Verified here on CPU in a fresh Python 3.12 virtualenv, using `uv` (plain `pip` should work the same way):

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[musegan,dev]"     # torch is only needed for MuseGAN and tests
pytest tests/test_musegan.py -q        # 7 shape/import tests; passed in ~14 s on CPU
python -c "from deepTechno.model.transformer import MusicTransformer; print('ok')"
```

What these commands do not cover:

- **Training** needs data and, realistically, a GPU. The transformer and text notebooks (`03`, `04`) download MAESTRO (about 57 MB) and write checkpoints to Google Drive. Training was not run here.
- **LSTM** needs `tensorflow`, which is not a declared dependency (commented out in `requirements.txt`). Install it yourself.
- **`all_notes.csv`** (about 950 MB, not in the repo) is produced or uploaded by `notebooks/00_setup_and_data.ipynb`.
- **Vision** extras: `pip install -e ".[vision]"` adds `timm`.

Run the notebooks in order, starting with [`00_setup_and_data.ipynb`](notebooks/00_setup_and_data.ipynb).

## Project structure

```
src/deepTechno/
  data/         MIDI parsing, tokenizer, MAESTRO dataset, CSV -> pianoroll bridge
  model/        MusicTransformer, relative-position attention, losses, constants
  encoders/     TextEncoder (used in notebook 04), VisionEncoder (placeholder)
  training/     LSTM and Transformer training loops
  generation/   sampling from LSTM / Transformer checkpoints
  musegan/      generator, critic, binary neuron, WGAN-GP trainer (scaffold)
  utils/        device selection, LR scheduling
notebooks/      00 setup, 01 explore, 02 LSTM, 03 Transformer, 04 text-conditioned, 05 vision
tests/          test_musegan.py (shape tests, no training)
results/        test.mid (sample MIDI)
_legacy/        earlier version of the code, kept for reference
CLAUDE.md       longer developer notes
```

## Results

No metrics, loss curves or plots are committed: the notebooks have no saved outputs. The only artifact is [`results/test.mid`](results/test.mid), a single-instrument MIDI file with 5,000 notes and about 582 s of audio. The repo does not record which model produced it.

## Tech stack

Python, PyTorch (Transformer, MuseGAN), TensorFlow/Keras (LSTM), pretty_midi, mido, pandas, matplotlib, Google Colab, uv/setuptools, pytest.

## Status

| Part | State |
|---|---|
| Music Transformer, notebook `03` | Implemented; no results in repo |
| Text conditioning, notebook `04` | Encoder and vocabulary implemented; no results in repo |
| LSTM, notebook `02` | Implemented; no results in repo |
| MuseGAN | Scaffold, shape-tested on CPU only. Needs GPU validation: training on real data, checking that losses are finite, tuning the pitch-register to track mapping and BPM, and adding evaluation metrics |
| Vision to MIDI (sheet music), notebook `05` | Placeholder encoder; cross-attention wiring not done. Targets the PrIMuS dataset |

## License

No license file is present in the repository.
