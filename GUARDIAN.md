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
| Lint | No ruff config in `pyproject.toml`. Baseline is `ruff check --isolated --select E4,E7,E9,F src tests` (ruff's default rule set) |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | high | bug (data loss) | src/deepTechno/data/tokenizer.py | fixed | 2026-10-15 |
| G-002 | medium | bug (data loss) | src/deepTechno/data/tokenizer.py | fixed | 2026-10-15 |
| G-003 | medium | bug (logic) | src/deepTechno/data/preprocess.py | fixed | 2026-10-15 |
| G-004 | medium | bug (logic) | src/deepTechno/data/preprocess.py | fixed | 2026-10-15 |
| G-005 | medium | vuln (B614 unsafe torch.load / pickle) | src/deepTechno/generation/generate_transformer.py | fixed | 2026-10-15 |
| G-006 | medium | vuln (B614 unsafe torch.load / pickle) | src/deepTechno/training/train_transformer.py | fixed | 2026-10-15 |
| G-007 | low | vuln (zip extractall) | src/deepTechno/data/sourcing.py | wontfix | 2026-10-15 |
| G-008 | low | bandit lows (B101 assert, B311 random, B105 false positives on `<pad>`-style tokens) | src/ | wontfix | 2026-10-15 |
| G-009 | low | quality (19 ruff default-rule errors: star import in rpr.py, E702, unused imports) | src/, tests/ | partially fixed (19 -> 3) | 2026-10-17 |
| G-010 | medium | bug (logic) | src/deepTechno/model/transformer.py | fixed | 2026-10-17 |
| G-011 | medium | bug (crash / invalid output) | src/deepTechno/data/tokenizer.py | fixed | 2026-10-17 |
| G-012 | low | bug (crash) | src/deepTechno/data/preprocess.py | fixed | 2026-10-17 |
| G-013 | low | bug (crash) | src/deepTechno/model/loss.py | fixed | 2026-10-17 |

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
- **G-007**: the archive is MAESTRO from a fixed Google URL, and `zipfile.extractall` strips `..` and absolute paths. Accepted as low risk.

## Metrics (before -> after)

| Metric | Before | After |
|---|---|---|
| Dependency vulns (crit/high/med/low) | 0/0/0/0 | 0/0/0/0 |
| Code vulns, bandit (high/med/low) | 0/0/13 | 0/0/13 |
| Secrets (gitleaks git + dir) | 0 | 0 |
| Failing tests | 0 | 0 |
| Tests passed | 12 | 17 |
| Known bugs open | 4 (2 from the backlog, 2 found this run) | 0 |
| Lint errors (ruff default rules) | 19 | 3 |
| Coverage (deepTechno) | 40% | 44% |

Run #1 (2026-10-15): bandit 0/2/13 -> 0/0/13, tests 7 -> 12, coverage 17% -> 40%.

## Backlog

- Random sampling in `MusicTransformer.generate` slices out TOKEN_END (`[..., :TOKEN_END]`), so the `next_token == TOKEN_END` stop branch never runs. This is inherited from upstream. Including END changes generation behaviour, so the owner should decide.
- `decode_midi` defaults velocity to 0 when no velocity token comes before a note. Velocity 0 is a MIDI note-off, so such notes vanish when the file is written.
- `SmoothCrossEntropyLoss` with `reduction="mean"` returns NaN when every target is ignored (division by 0).
- `CLAUDE.md` says `VOCAB_SIZE = 388`, but the code has 390 (388 events + END + PAD). The code is correct and the doc is out of date.
- G-009 leftovers: E731 in `sourcing.py`, and E402 in `tests/test_musegan.py` (add a per-file ignore once a ruff config exists).
- Add CI (GitHub Actions with pinned SHAs and `permissions: contents: read`) that runs the pytest suite with torch installed.
- TensorFlow-dependent modules (`sourcing.py`, `train_lstm.py`, `generate_lstm.py`) are untested here.

## Learnings

- Without torch, the existing suite reports "1 skipped" (`pytest.importorskip`), which hides that nothing ran. Install torch to get the real baseline.
- `download.pytorch.org` is blocked by the proxy. The PyPI torch wheel installs and runs on CPU.
- osv.dev API calls return 403 through the proxy, so pip-audit against `uv export` output is the reliable dependency scanner here.
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
