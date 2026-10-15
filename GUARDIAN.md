# GUARDIAN

Automated security and bug-fix ledger for `deep-techno`. Branch: `claude/guardian`.

## Checks

| Check | Command |
|---|---|
| Dependency CVEs (resolved lockfile, all extras) | `uv export --frozen --no-hashes --all-extras > req.txt && pip-audit -r req.txt --no-deps --disable-pip` |
| OSV | `osv-scanner -r .` (the osv.dev API returned 403 through the proxy, so this gave no result; `--lockfile uv.lock` is not supported by this osv-scanner version) |
| SAST | `bandit -r src` (all severities), then `bandit -r src -ll` (medium and above) |
| Pattern grep | `grep -rnE "eval\(|exec\(|pickle|torch\.load|joblib\.load|yaml\.load|shell=True|verify=False|0\.0\.0\.0|subprocess|os\.system|extractall" src` |
| Secrets | `gitleaks git . --redact` (14 commits, full history) and `gitleaks dir . --redact` |
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
| G-009 | low | quality (19 ruff default-rule errors: star import in rpr.py, E702, unused imports) | src/, tests/ | open | 2026-10-15 |

Details:
- **G-001**: `_note_preprocess` lost upstream's no-sustain branch. Any MIDI without CC64 pedal events (all techno and drum MIDI) encoded to `[]`, so `MaestroDataset` produced sequences made only of PAD tokens.
- **G-002**: notes after the last sustain-pedal release were silently dropped. Once a sustain had used up every remaining note, the next sustain went over those same notes again and duplicated them.
- **G-003**: `csv_to_pianoroll_dataset` sorted all of `all_notes.csv` by `start`. Each piece's clock restarts at 0, so that sort interleaved unrelated pieces into the same window. Pieces are now split wherever `start` drops.
- **G-004**: each window was re-anchored to its first note rather than to the window start. That broke bar alignment and dropped notes at the end of the window. A new `origin=` parameter fixes this.
- **G-005/G-006**: `torch.load` is now called with `weights_only=True`. torch 2.6+ already does this by default, but `pyproject.toml` allows `torch>=2.2`. The training resume path also gets `map_location`.
- **G-007**: the archive is MAESTRO from a fixed Google URL, and `zipfile.extractall` strips `..` and absolute paths. Accepted as low risk.

## Metrics (before -> after)

| Metric | Before | After |
|---|---|---|
| Dependency vulns (crit/high/med/low) | 0/0/0/0 | 0/0/0/0 |
| Code vulns, bandit (high/med/low) | 0/2/13 | 0/0/13 |
| Secrets (gitleaks git + dir) | 0 | 0 |
| Failing tests | 0 | 0 |
| Tests passed | 7 | 12 |
| Known bugs open | 4 (uncovered) | 0 |
| Lint errors (ruff default rules) | 19 | 19 |
| Coverage (deepTechno) | 17% | 40% |

## Backlog

- `midi_to_notes` raises `IndexError` on a MIDI with no instruments or no notes.
- `SmoothCrossEntropyLoss` with the default `ignore_index=-100` crashes in `F.one_hot`. It is currently only used with `TOKEN_PAD`.
- `CLAUDE.md` says `VOCAB_SIZE = 388`, but the code has 390 (388 events + END + PAD). The code is correct and the doc is out of date.
- G-009 lint cleanup, starting with replacing the star import in `rpr.py` with explicit imports.
- Add CI (GitHub Actions with pinned SHAs and `permissions: contents: read`) that runs the pytest suite with torch installed.
- TensorFlow-dependent modules (`sourcing.py`, `train_lstm.py`, `generate_lstm.py`) are untested here.

## Learnings

- Without torch, the existing suite reports "1 skipped" (`pytest.importorskip`), which hides that nothing ran. Install torch to get the real baseline.
- `download.pytorch.org` is blocked by the proxy. The PyPI torch wheel installs and runs on CPU.
- osv.dev API calls return 403 through the proxy, so pip-audit against `uv export` output is the reliable dependency scanner here.

## Ideas

1. Write a minimal GitHub Actions workflow with pytest (CPU torch), bandit and gitleaks.
2. Add a round-trip property test for `encode_midi`/`decode_midi` over random note sets.
3. Have `midi_to_notes` emit a `piece_id` column, so the MuseGAN bridge does not have to infer piece boundaries.

## Log

- 2026-10-15 run #1 — First run. All scanners were clean for dependencies and secrets. Fixed 4 data-pipeline bugs (tokenizer drops all notes when there is no sustain pedal; post-sustain notes lost; pianoroll windows mixed pieces; window grid not anchored) and 2 medium bandit findings (`torch.load` without `weights_only`). Added 5 tests: 7 -> 12 passing, coverage 17% -> 40%.
