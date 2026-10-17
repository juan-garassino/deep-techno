# GUARDIAN

Automated security and bug-fix ledger for `deep-techno`. Branch: `claude/guardian`.

## Checks

| Check | Command |
|---|---|
| Dependency CVEs (resolved lockfile, all extras) | `uv export --frozen --no-hashes --all-extras > req.txt && pip-audit -r req.txt --no-deps --disable-pip` |
| OSV | `osv-scanner -r .` (the osv.dev API returned 403 through the proxy, so this gave no result; `--lockfile uv.lock` is not supported by this osv-scanner version) |
| SAST | `bandit -r src` (all severities), then `bandit -r src -ll` (medium and above) |
| Pattern grep | `grep -rnE "eval\(|exec\(|pickle|torch\.load|joblib\.load|yaml\.load|shell=True|verify=False|0\.0\.0\.0|subprocess|os\.system|extractall" src` |
| Secrets | `gitleaks git . --redact` (full history) and `gitleaks dir . --redact` (gitleaks v8.21.2 release tarball from GitHub, no package manager needed) |
| CI / containers | No `.github/workflows` and no `Dockerfile` in the repo, so there was nothing to check |
| Tests | `pytest -q --cov=deepTechno` in a uv venv (py3.12, `-e .[dev]`, torch 2.11 from PyPI; the CPU wheel index was unreachable; TensorFlow and timm were skipped to save time) |
| Lint | No ruff config in `pyproject.toml`. Baseline is `ruff check --isolated --select E4,E7,E9,F src tests` (ruff's default rule set). It has been clean since run #2, so any new error is a regression |
| Torch API drift | `pytest -q -k "forward_on_current_torch or training_smoke"`. It runs a real `MusicTransformer` forward (RPR and non-RPR) and a 1-epoch train + resume on the newest torch the lock allows. Upstream's custom `nn.Transformer` encoder/decoder broke silently when torch 2.1 started passing new kwargs (G-014) |
| Archive extraction | `grep -rn "extractall\|tarfile" src`. Every hit must go through `sourcing.safe_extract` or an equivalent member check |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | high | bug (data loss) | src/deepTechno/data/tokenizer.py | fixed | 2026-10-15 |
| G-002 | medium | bug (data loss) | src/deepTechno/data/tokenizer.py | fixed | 2026-10-15 |
| G-003 | medium | bug (logic) | src/deepTechno/data/preprocess.py | fixed | 2026-10-15 |
| G-004 | medium | bug (logic) | src/deepTechno/data/preprocess.py | fixed | 2026-10-15 |
| G-005 | medium | vuln (B614 unsafe torch.load / pickle) | src/deepTechno/generation/generate_transformer.py | fixed | 2026-10-15 |
| G-006 | medium | vuln (B614 unsafe torch.load / pickle) | src/deepTechno/training/train_transformer.py | fixed | 2026-10-15 |
| G-007 | low | vuln (zip extractall / zip-slip) | src/deepTechno/data/sourcing.py | fixed (run #2 pass 2: `safe_extract`) | 2026-10-17 |
| G-008 | low | bandit lows (B101 assert, B311 random, B105 false positives on `<pad>`-style tokens) | src/ | wontfix | 2026-10-15 |
| G-009 | low | quality (19 ruff default-rule errors: star import in rpr.py, E702, unused imports) | src/, tests/ | fixed (19 -> 0) | 2026-10-17 |
| G-010 | medium | bug (logic) | src/deepTechno/model/transformer.py | fixed | 2026-10-17 |
| G-011 | medium | bug (crash / invalid output) | src/deepTechno/data/tokenizer.py | fixed | 2026-10-17 |
| G-012 | low | bug (crash) | src/deepTechno/data/preprocess.py | fixed | 2026-10-17 |
| G-013 | low | bug (crash) | src/deepTechno/model/loss.py | fixed | 2026-10-17 |
| G-014 | high | bug (crash: every MusicTransformer forward) | src/deepTechno/model/transformer.py, rpr.py | fixed | 2026-10-17 |
| G-015 | low | bug (tokenization) | src/deepTechno/encoders/text_encoder.py | fixed | 2026-10-17 |
| G-016 | low | bug (NaN loss) | src/deepTechno/model/loss.py | fixed | 2026-10-17 |
| G-017 | low | bug (crash on resume) | src/deepTechno/training/train_transformer.py | fixed | 2026-10-17 |

Details:
- **G-001**: `_note_preprocess` lost upstream's no-sustain branch. Any MIDI without CC64 pedal events (all techno and drum MIDI) encoded to `[]`, so `MaestroDataset` produced sequences made only of PAD tokens.
- **G-002**: notes after the last sustain-pedal release were silently dropped. Once a sustain had used up every remaining note, the next sustain went over those same notes again and duplicated them.
- **G-003**: `csv_to_pianoroll_dataset` sorted all of `all_notes.csv` by `start`. Each piece's clock restarts at 0, so that sort interleaved unrelated pieces into the same window. Pieces are now split wherever `start` drops.
- **G-004**: each window was re-anchored to its first note rather than to the window start. That broke bar alignment and dropped notes at the end of the window. A new `origin=` parameter fixes this.
- **G-005/G-006**: `torch.load` is now called with `weights_only=True`. torch 2.6+ already does this by default, but `pyproject.toml` allows `torch>=2.2`. The training resume path also gets `map_location`.
- **G-009** (run #2): replaced the star import in `rpr.py` with explicit imports, removed 5 unused imports and split the E702 one-liners. What is left: E731 in `sourcing.py` (TF module, untested here) and E402 in `tests/test_musegan.py` (needed for the `importorskip` pattern).
- **G-010**: in beam search, `MusicTransformer.generate` slices the probabilities to `TOKEN_END` (388) columns, but it split the flat top-k index by `VOCAB_SIZE` (390). From the second step on, this picked the wrong beam rows and shifted tokens by 2 per row, which could land on END/PAD. It now uses the real row width. Test: `test_beam_search_picks_correct_rows_and_tokens`.
- **G-011**: `decode_midi` treated END (388) and PAD (389) as velocity events with velocity 128 or 132. `mido` then raised `ValueError: data byte must be in range 0..127` on write. This was reachable from `generate_from_dataset`, because a short primer is END/PAD-padded. Decoding now stops at END and skips PAD. Tests: `test_decode_midi_ignores_end_and_pad_tokens`, `test_decode_midi_stops_at_end_token`.
- **G-012**: `midi_to_notes` raised `IndexError` on a MIDI with no instruments or no notes. It now returns an empty frame with the 5 columns. Test: `test_midi_to_notes_empty_midi_returns_empty_frame`.
- **G-013**: `SmoothCrossEntropyLoss` with the default `ignore_index=-100` crashed in `F.one_hot`. Ignored targets are now masked to 0 before one-hot. Test: `test_smooth_ce_default_ignore_index`.
- **G-007**: the archive is MAESTRO from a fixed Google URL, and `zipfile.extractall` already strips `..` and absolute paths. Run #2 pass 2 added `safe_extract`, which checks every member before writing anything (defence in depth). `sourcing.py` now also imports TensorFlow lazily, so its zip and CSV helpers can be tested without TF. Tests: `test_safe_extract_rejects_zip_slip`, `test_safe_extract_extracts_normal_members` and `test_find_midi_files_and_split_csv`.
- **G-014**: since torch 2.1, `nn.Transformer.forward` passes `tgt_is_causal`/`memory_is_causal` to the decoder and `is_causal` to the encoder. `DummyDecoder` and `TransformerEncoderRPR` had fixed signatures, so with `torch>=2.2` (the project's own pin) every `MusicTransformer` forward raised `TypeError`. That broke training, eval and generation, both RPR and non-RPR. Existing tests missed it because the MuseGAN tests never touch this model and run #1's beam test mocked `forward`. Tests: `test_music_transformer_forward_on_current_torch[False/True]`, `test_generate_from_dataset_short_primer_writes_midi` and `test_transformer_training_smoke_and_resume_without_epoch`.
- **G-015**: `_tokenize` used `[a-z0-9]+`, which split accented MAESTRO names into fragments, e.g. "Frédéric" -> `fr`, `d`, `ric`. Composer-conditioning tokens were therefore wrong. Test: `test_tokenize_keeps_accented_words`. Any vocab JSON saved before this fix should be rebuilt.
- **G-016**: `SmoothCrossEntropyLoss(reduction="mean")` returned NaN (0/0) for an all-ignored batch, which an empty MIDI produces (all PAD). Test: `test_smooth_ce_all_targets_ignored_is_finite`.
- **G-017**: `run_transformer_training` with `continue_weights` but no `continue_epoch` ran `range(None, ...)` and raised `TypeError`. It now resumes from epoch 0. Test: `test_transformer_training_smoke_and_resume_without_epoch`.

## Metrics (before -> after)

| Metric | Before | After |
|---|---|---|
Run #2 (2026-10-17), cumulative over pass 1 and pass 2:

| Metric | Before | After |
|---|---|---|
| Dependency vulns (crit/high/med/low) | 0/0/0/0 | 0/0/0/0 |
| Code vulns, bandit (high/med/low) | 0/0/13 | 0/0/13 (all wontfix false positives, G-008) |
| Ledger vulns open (high/med/low) | 0/0/1 (G-007) | 0/0/0 |
| Secrets (gitleaks git + dir) | 0 | 0 |
| Failing tests | 0 | 0 |
| Tests passed | 12 | 27 |
| Known bugs open | 9 (found this run) | 0 |
| Lint errors (ruff default rules) | 19 | 0 |
| Coverage (deepTechno) | 40% | 77% |

Run #1 (2026-10-15): bandit 0/2/13 -> 0/0/13, tests 7 -> 12, coverage 17% -> 40%.
The run #2 table lists only the metrics recorded in run #2.

## Backlog

- Random sampling in `MusicTransformer.generate` slices out TOKEN_END (`[..., :TOKEN_END]`), so the `next_token == TOKEN_END` stop branch never runs. This is inherited from upstream. Including END changes generation behaviour, so the owner should decide.
- `decode_midi` defaults velocity to 0 when no velocity token comes before a note. Velocity 0 is a MIDI note-off, so such notes vanish when the file is written.
- `CLAUDE.md` says `VOCAB_SIZE = 388`, but the code has 390 (388 events + END + PAD). The code is correct and the doc is out of date.
- `TextEncoder.forward` raises IndexError when the input is longer than `max_text_len`, because nothing truncates it or checks it. `TextVocab.encode(max_len < 2)` slices the input oddly.
- `MaestroDataset.__getitem__` swallows every encode exception (`except Exception`), so a corrupt or missing file silently becomes an all-PAD sample. At least log it.
- `generate_from_dataset` gives `generate` a primer that contains END/PAD, so the model is conditioned on padding. Strip these from the primer (Ideas #6).
- `eval_model` divides by `len(dataloader)`, so an empty validation CSV raises ZeroDivisionError.
- Add CI (GitHub Actions with pinned SHAs and `permissions: contents: read`) that runs the pytest suite with torch installed.
- TensorFlow-dependent modules (`sourcing.py`, `train_lstm.py`, `generate_lstm.py`) are untested here.

## Learnings

- Without torch, the existing suite reports "1 skipped" (`pytest.importorskip`), which hides that nothing ran. Install torch to get the real baseline.
- `download.pytorch.org` is blocked by the proxy. The PyPI torch wheel installs and runs on CPU.
- osv.dev API calls return 403 through the proxy, so pip-audit against `uv export` output is the reliable dependency scanner here.
- Mocked-forward tests can hide a broken real forward: run #1's beam test passed while every real `MusicTransformer.forward` crashed (G-014). Keep at least one unmocked tiny-model test for each public model path.
- Check that each new regression test fails on the unfixed code (`git stash`, run the test, `git stash pop`). The first draft of the decode test passed on buggy code because `encode_midi` emits its own velocity token.
- `tokenizer.py` is deliberately torch-free (`model/constants.py` imports torch), so tokenizer-side constants are mirrored locally rather than imported.

## Ideas

1. Write a minimal GitHub Actions workflow with pytest (CPU torch), bandit and gitleaks.
2. Add a round-trip property test for `encode_midi`/`decode_midi` over random note sets.
3. Have `midi_to_notes` emit a `piece_id` column, so the MuseGAN bridge does not have to infer piece boundaries.
4. Add a `[tool.ruff]` section to `pyproject.toml` so the lint baseline is reproducible without `--isolated`.
5. Add a smoke test for `generate_from_dataset` with a tiny model and a short MIDI, covering the full primer -> generate -> decode -> write path.
6. Have `MusicTransformer.generate` validate that the primer has no PAD tokens, or strip trailing END/PAD from dataset primers.

## Log

- 2026-10-15 run #1 — First run. All scanners were clean for dependencies and secrets. Fixed 4 data-pipeline bugs (tokenizer drops all notes when there is no sustain pedal; post-sustain notes lost; pianoroll windows mixed pieces; window grid not anchored) and 2 medium bandit findings (`torch.load` without `weights_only`). Added 5 tests: 7 -> 12 passing, coverage 17% -> 40%.
- 2026-10-17 run #2 — No open PR review comments. Dependencies (pip-audit), secrets (gitleaks, history and tree) and bandit were all clean, with no new findings. Fixed 4 bugs, each with a test that fails without the fix: beam-search row/token indexing (G-010), END/PAD tokens decoded as invalid velocities (G-011), `midi_to_notes` IndexError on empty MIDI (G-012) and the `SmoothCrossEntropyLoss` default `ignore_index` crash (G-013). Lint went from 19 to 3. Tests went from 12 to 17 passing and coverage from 40% to 44%. Not a meta-review run (2 of 4).
- 2026-10-17 run #2 pass 2 — Found that every `MusicTransformer` forward crashed on `torch>=2.2`: `DummyDecoder` and the RPR encoder rejected the new causal kwargs (G-014, high). Fixed it and added unmocked forward, generation and train+resume smoke tests. Also added zip-slip-safe extraction with a lazy TF import (G-007 closed), kept accented composer names whole in the text tokenizer (G-015), fixed the all-ignored NaN loss (G-016) and the resume without `continue_epoch` TypeError (G-017). Lint went from 3 to 0, tests from 17 to 27 and coverage from 44% to 77%. Added the Torch-API-drift and archive-extraction checks.
