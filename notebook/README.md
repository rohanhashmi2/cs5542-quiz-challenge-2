# Notebook

This directory contains the end-to-end Colab notebook (`quiz_challenge_2.ipynb`) that produced every result in `results/` and `spectrograms/`.

It is the **source of truth** for this project. The `src/` package is a clean, importable reorganization of the same code — see [`cell_map.md`](cell_map.md) for a function-by-function traceability map.

## How to run

1. Upload `quiz_challenge_2.ipynb` to Google Colab.
2. `Runtime → Change runtime type → T4 GPU` (or L4 / A100).
3. Run cells top-to-bottom.

Expected runtime on NVIDIA L4: **~15 minutes end-to-end** (dependencies + model load + 45-clip generation + 5 metrics).

## What each cell does

- Cells 1–7: Dependency install (with numpy pinning for librosa compatibility), GPU verification, MusicGen load, test generation
- Cells 8–9: Spectrogram utility functions
- Cells 10–13: Scene schema (`SceneSpec` + 5-scene catalog), three prompt strategies
- Cells 14–15: Single-scene preview across three strategies
- Cells 16–18: Full 45-clip generation suite + BPM preview
- Cells 19–20: Load all clips into memory for evaluation
- Cells 21–22: Load CLAP model for audio-text alignment scoring
- Cells 23–32: Compute five metrics — raw CLAP, semantic CLAP, consistency, diversity, tempo accuracy
- Cells 33–34: Final summary table + CSV/JSON export

## Dependency gotcha

Colab ships with numpy 2.x by default, but `librosa` requires numpy < 2.0 through its `numba` dependency. The notebook pins numpy explicitly in Cell 1:

```python
!pip install -q "numpy>=1.26,<2.0"
```

After this pin, **restart the Colab runtime** once to clear cached imports. All subsequent cells work normally.
